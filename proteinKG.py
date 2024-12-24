import argparse
import asyncio
import json
import logging
import os
import os.path as osp
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from rich.logging import RichHandler
from openai import AsyncOpenAI

import ollama
from nano_graphrag._storage import (
    NetworkXStorage,
    NanoVectorDBStorage,
    SeqVectorDBStorage,
    JsonKVStorage
)
from nano_graphrag._utils import (
    EmbeddingFunc, 
    compute_mdhash_id, 
    limit_async_func_call, 
    wrap_embedding_func_with_attrs,
    compute_args_hash,
    convert_response_to_json
)
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import logger

PROTEIN_SYS = "PROTEIN_{id}"
GO_SYS = "GO_{id}"

### Prepare LLM
WORKING_DIR = "./proteinKG_cache"
# Assumed llm model settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_API_KEY")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


async def llm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


class ProteinKGConfig(NamedTuple):
    working_dir: str = WORKING_DIR
    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 500
    graph_cluster_seed: int = 0xDEADBEEF
    graph_cluster_resolution: float = 2.0
    
    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = {
        "dimensions": 1536,
        "num_walks": 10,
        "walk_length": 40,
        "num_walks": 10,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }

class ProteinVectorDBConfig(NamedTuple):
    working_dir: str = WORKING_DIR
    esmc_model: str = "esmc_300m"
    device: str = "cuda"
    embedding_batch_num: int = 1
    embedding_dim: int = 960
    embedding_func_max_async: int = 1
    query_better_than_threshold: float = 0.2

class ProteinKGCommunityConfig(NamedTuple):
    working_dir: str = WORKING_DIR
    # community reports
    special_community_report_llm_kwargs: dict = {"response_format": {"type": "json_object"}}

class ProteinKGLLMConfig(NamedTuple):
    # LLM
    working_dir: str = WORKING_DIR
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = llm_model_if_cache
    best_model_max_token_size: int = 12000
    best_model_max_async: int = 8
    cheap_model_func: callable = llm_model_if_cache
    cheap_model_max_token_size: int = 12000
    cheap_model_max_async: int = 8

class ProteinKGExtensionConfig(NamedTuple):
    # extension
    always_create_working_dir: bool = True
    addon_params: dict = {}
    convert_response_to_json_func: callable = convert_response_to_json

def prepare_kg_global_config():
    
    logger.info("Preparing global config")
    kg_global_config = ProteinKGConfig()
    if not osp.exists(kg_global_config.working_dir):
        os.makedirs(kg_global_config.working_dir, exist_ok=True)

    vdb_global_config = ProteinVectorDBConfig()
    
    community_global_config = ProteinKGCommunityConfig()
    if not osp.exists(community_global_config.working_dir):
        os.makedirs(community_global_config.working_dir, exist_ok=True)

    llm_global_config = ProteinKGLLMConfig()
    extension_global_config = ProteinKGExtensionConfig()
    
    global_config = {
        **kg_global_config._asdict(),
        **vdb_global_config._asdict(),
        **community_global_config._asdict(),
        **llm_global_config._asdict(),
        **extension_global_config._asdict(),
    }
    return global_config

global_config = prepare_kg_global_config()

async def upsert_protein_kg():
    
    ### Prepare KG DATA
    ROOT_DIR = Path("./ProteinKG25")

    logger.info("Preparing KG files")
    file_list = [
        'go2id.txt',
        'protein_go_train_triplet.txt', 
        'relation2id.txt', 
        'go_go_triplet.txt', 
        'go_def.txt', 
        'protein2id.txt', 
        'protein_seq.txt', 
        'protein_go_test_triplet.txt', 
        'train2id.txt', 
        'go_type.txt', 
        'protein_go_valid_triplet.txt'
    ]
    # Check if the data is already prepared
    for file in file_list:
        if not osp.exists(osp.join(ROOT_DIR, file)):
            logger.error(f"KG data is not prepared, missing file: {file}")
            return

    GO_ID2CLASS = {}
    with open(osp.join(ROOT_DIR, "go2id.txt"), "r") as f:
        for line in f:
            go_name, go_id = line.strip().split(" ")
            GO_ID2CLASS[int(go_id)] = go_name

    GO_CLASS2ID = {v: k for k, v in GO_ID2CLASS.items()}
    
    with open(osp.join(WORKING_DIR, "go_name2id.json"), "w") as f:
        f.write(json.dumps(GO_CLASS2ID, indent=4))

    PROTEIN_ID2NAME = {}
    with open(osp.join(ROOT_DIR, "protein2id.txt"), "r") as f:
        for line in f:
            protein_name, protein_id = line.strip().split(" ")
            PROTEIN_ID2NAME[int(protein_id)] = protein_name

    PROTEIN_NAME2ID = {v: k for k, v in PROTEIN_ID2NAME.items()}
    
    with open(osp.join(WORKING_DIR, "protein_name2id.json"), "w") as f:
        f.write(json.dumps(PROTEIN_NAME2ID, indent=4))
        
    PROTEIN_SEQ2ID = defaultdict(list)
    with open(osp.join(ROOT_DIR, "protein_seq.txt"), 'r') as f:
        for i, line in enumerate(f):
            PROTEIN_SEQ2ID[line.strip()].append(i)

    PROTEIN_ID2SEQ = {i: seq for seq, ids in PROTEIN_SEQ2ID.items() for i in ids}
    
    with open(osp.join(WORKING_DIR, "protein_seq2id.json"), "w") as f:
        f.write(json.dumps(PROTEIN_SEQ2ID, indent=4))

    RELATION_ID2NAME = {}
    with open(osp.join(ROOT_DIR, "relation2id.txt"), "r") as f:
        for line in f:
            relation_name, relation_id = line.strip().split("\t")
            RELATION_ID2NAME[int(relation_id)] = relation_name

    RELATION_NAME2ID = {v: k for k, v in RELATION_ID2NAME.items()}
    
    with open(osp.join(WORKING_DIR, "relation_name2id.json"), "w") as f:
        f.write(json.dumps(RELATION_NAME2ID, indent=4))
        
    GO_DEF = {}
    with open(osp.join(ROOT_DIR, "go_def.txt"), "r") as f:
        for i, line in enumerate(f):
            go_line = line.strip().split(": ")
            go_name = go_line[0]
            go_def = ": ".join(go_line[1:])
            GO_DEF[i] = {
                "name": go_name,
                "definition": go_def
            }

    GO_TYPE = {}
    with open(osp.join(ROOT_DIR, "go_type.txt"), "r") as f:
        for i, line in enumerate(f):
            go_type = line.strip()
            GO_TYPE[i] = go_type

    GO_GO_TRIPLET = []
    with open(osp.join(ROOT_DIR, "go_go_triplet.txt"), "r") as f:
        for line in f:
            go_a, relation, go_b = line.strip().split(" ")
            GO_GO_TRIPLET.append((int(go_a), int(relation), int(go_b)))

    PROTEIN_GO_TRIPLET = []
    with open(osp.join(ROOT_DIR, "protein_go_train_triplet.txt"), "r") as f:
        for line in f:
            protein_id, relation, go_id = line.strip().split(" ")
            PROTEIN_GO_TRIPLET.append((int(protein_id), int(relation), int(go_id)))
    with open(osp.join(ROOT_DIR, "protein_go_valid_triplet.txt"), "r") as f:
        for line in f:
            protein_id, relation, go_id = line.strip().split(" ")
            PROTEIN_GO_TRIPLET.append((int(protein_id), int(relation), int(go_id)))
    with open(osp.join(ROOT_DIR, "protein_go_test_triplet.txt"), "r") as f:
        for line in f:
            protein_id, relation, go_id = line.strip().split(" ")
            PROTEIN_GO_TRIPLET.append((int(protein_id), int(relation), int(go_id)))
            
            

    PROTEIN_SEQ2ID = defaultdict(list)
    with open(osp.join(ROOT_DIR, "protein_seq.txt"), 'r') as f:
        for i, line in enumerate(f):
            PROTEIN_SEQ2ID[line.strip()].append(i)

    PROTEIN_ID2SEQ = {i: seq for seq, ids in PROTEIN_SEQ2ID.items() for i in ids}
     
     

    PROTEIN_SEQ2ID = defaultdict(list)
    with open(osp.join(ROOT_DIR, "protein_seq.txt"), 'r') as f:
        for i, line in enumerate(f):
            PROTEIN_SEQ2ID[line.strip()].append(i)

    PROTEIN_ID2SEQ = {i: seq for seq, ids in PROTEIN_SEQ2ID.items() for i in ids}
     
    logger.info("Preparing KG")
    knwoledge_graph_inst = NetworkXStorage(
        namespace="protein_kg",
        global_config=global_config
    )

    nodes_data: list[dict] = []
    edges_data: list[dict] = []
    for go_a, relation, go_b in GO_GO_TRIPLET:
        nodes_data.append({
            "id": f"GO_{go_a}",
            "entity_attr": "GO",
            "entity_name": f"{GO_ID2CLASS[go_a]}",
            "description": f"{GO_DEF[go_a]['name']} [{GO_TYPE[go_a]}]: {GO_DEF[go_a]['definition']}",
        })
        nodes_data.append({
            "id": f"GO_{go_b}",
            "entity_attr": "GO",
            "entity_name": f"{GO_ID2CLASS[go_b]}",
            "description": f"{GO_DEF[go_b]['name']} [{GO_TYPE[go_b]}]: {GO_DEF[go_b]['definition']}",
        })
        edges_data.append({
            "src_id": f"GO_{go_a}",
            "tgt_id": f"GO_{go_b}",
            "src_name": f"{GO_ID2CLASS[go_a]}",
            "tgt_name": f"{GO_ID2CLASS[go_b]}",
            "description": f"{RELATION_ID2NAME[relation]}",
        })
    
    for protein_id, relation, go_id in PROTEIN_GO_TRIPLET:
        nodes_data.append({
            "id": f"PROTEIN_{protein_id}",
            "entity_attr": "PROTEIN",
            "entity_name": f"{PROTEIN_ID2NAME[protein_id]}",
            "description": f"{PROTEIN_ID2SEQ[protein_id]}",
        })
        edges_data.append({
            "src_id": f"PROTEIN_{protein_id}",
            "tgt_id": f"GO_{go_id}",
            "src_name": f"{PROTEIN_ID2NAME[protein_id]}",
            "tgt_name": f"{GO_ID2CLASS[go_id]}",
            "description": f"{RELATION_ID2NAME[relation]}",
        })

    all_entities_data: list[dict] = []
    for node_data in tqdm(nodes_data, desc="Upserting nodes"):
        node_id = node_data.pop("id")
        await knwoledge_graph_inst.upsert_node(
            node_id,
            node_data=node_data,
        )
        all_entities_data.append({
            "id": node_id,
            **node_data
        })

    for edge_data in tqdm(edges_data, desc="Upserting edges"):
        await knwoledge_graph_inst.upsert_edge(
            edge_data.pop("src_id"),
            edge_data.pop("tgt_id"),
            edge_data=edge_data,
        )
    
    del nodes_data, edges_data
    
    # TODO: Optional
    # if not osp.exists(knwoledge_graph_inst._graphml_xml_file):
    #     await knwoledge_graph_inst.index_done_callback()
    
    # ! Save the graph to pickle file will accelerate the loading
    with open(osp.join(global_config["working_dir"], f"graph_protein_kg.pkl"), "wb") as f:
        pickle.dump(knwoledge_graph_inst, f)
    logger.info("Upserted nodes and edges")


async def search_seqdb(query: str):
    
    seqdb = SeqVectorDBStorage(
        namespace="protein_seqdb",
        global_config=global_config
    )
    logger.info(f"Searching in the seqdb with query: {query}")
    results = await seqdb.query(query)
    return results

async def generate_seqdb_embedding():
    
    # Load the graph from pickle file
    kg_fpath = osp.join(global_config["working_dir"], f"graph_protein_kg.pkl")
    with open(kg_fpath, "rb") as f:
        logger.info(f"Loading graph from pickle file: {kg_fpath}")
        knwoledge_graph_inst: NetworkXStorage = pickle.load(f)
    logger.info(knwoledge_graph_inst._graph)
    
    seqs_dict = {}
    for node, node_data in knwoledge_graph_inst._graph.nodes(data=True):
        if node.startswith("PROTEIN_"):
            seqs_dict[node] = node_data["description"]
    
    seqdb = SeqVectorDBStorage(
        namespace="protein_seqdb",
        global_config=global_config
    )
    await seqdb.upsert(seqs_dict)
    await seqdb.index_done_callback()
    

async def shortest_path(protein_node_id: str, go_node_id: str):
    
    # Load the graph from pickle file
    kg_fpath = osp.join(global_config["working_dir"], f"graph_protein_kg.pkl")
    with open(kg_fpath, "rb") as f:
        logger.info(f"Loading graph from pickle file: {kg_fpath}")
        knwoledge_graph_inst: NetworkXStorage = pickle.load(f)
    logger.info(knwoledge_graph_inst._graph)
    path = nx.shortest_path(knwoledge_graph_inst._graph, protein_node_id, go_node_id)
    logger.info(path)
    
    entity_fpath = osp.join(
        global_config["working_dir"], 
        f"{protein_node_id}-{go_node_id}",
        f"entity.csv"
    )
    edge_fpath = osp.join(
        global_config["working_dir"], 
        f"{protein_node_id}-{go_node_id}",
        f"relation.csv"
    )
    
    # Detect the entity and edge csv file
    if osp.exists(entity_fpath) and osp.exists(edge_fpath):
        logger.info(f"Entity and edge csv files already exist ")
        logger.info(f"Skip generating and using cached csv files")
        return
    
    entity_df = pd.DataFrame(columns=["entity_attr", "entity_name", "description"])
    edge_df = pd.DataFrame(columns=["source", "target", "description"])
    
    for node in path:
        node_data = await knwoledge_graph_inst.get_node(node)
        entity_df.loc[len(entity_df)] = [
            node_data["entity_attr"],
            node_data["entity_name"],
            node_data["description"]
        ]
        for src, tgt in zip(path[:-1], path[1:]):
            edge_data = await knwoledge_graph_inst.get_edge(src, tgt)
            src_node_data = await knwoledge_graph_inst.get_node(src)
            tgt_node_data = await knwoledge_graph_inst.get_node(tgt)
            (
                src_name,
                src_attr,
                tgt_name,
                tgt_attr,
                edge_description
            ) = (
                src_node_data["entity_name"],
                src_node_data["entity_attr"],
                tgt_node_data["entity_name"],
                tgt_node_data["entity_attr"],
                edge_data["description"]
            )
            edge_df.loc[len(edge_df)] = [
                src_name, tgt_name, 
                f"{src_name} ({src_attr}) {edge_description} {tgt_name} ({tgt_attr})"
            ]
    
    # Drop duplicate rows
    entity_df = entity_df.drop_duplicates()
    edge_df = edge_df.drop_duplicates()
    logger.info(entity_df)
    logger.info(edge_df)
    
    os.makedirs(osp.dirname(entity_fpath), exist_ok=True)
    entity_df.to_csv(entity_fpath, index=False)
    os.makedirs(osp.dirname(edge_fpath), exist_ok=True)
    edge_df.to_csv(edge_fpath, index=False)


async def generate_kg_report(protein_node_id: str, go_node_id: str):
    
    entity_fpath = osp.join(
        global_config["working_dir"], 
        f"{protein_node_id}-{go_node_id}",
        f"entity.csv"
    )
    relation_fpath = osp.join(
        global_config["working_dir"], 
        f"{protein_node_id}-{go_node_id}",
        f"relation.csv"
    )
    with open(entity_fpath, "r") as f:
        entity_df = f.read()
    with open(relation_fpath, "r") as f:
        relation_df = f.read()
        
    # convert to prompt format
    entity_text = f"Entities:\n{entity_df}\n"
    relation_text = f"Relations:\n{relation_df}\n"
    describe = f"{entity_text}\n{relation_text}"

    this_global_config = global_config.copy()
    this_global_config["working_dir"] = osp.join(
        this_global_config["working_dir"], 
        f"{protein_node_id}-{go_node_id}"
    )
    llm_response_cache = JsonKVStorage(
        namespace="kg_report_llm_response_cache",
        global_config=this_global_config
    )
    summary_cache = JsonKVStorage(
        namespace="kg_report_summary_cache",
        global_config=this_global_config
    )
    this_global_config["best_model_func"] = limit_async_func_call(
        this_global_config["best_model_max_async"]
    )(
        partial(
            this_global_config["best_model_func"], 
            hashing_kv=llm_response_cache
        )
    )
    this_global_config["cheap_model_func"] = limit_async_func_call(
        this_global_config["cheap_model_max_async"]
    )(
        partial(
            this_global_config["cheap_model_func"], 
            hashing_kv=llm_response_cache
        )
    )
    use_llm_func: callable = this_global_config["best_model_func"]
    use_string_json_convert_func: callable = this_global_config["convert_response_to_json_func"]
    kg_report_prompt = PROMPTS["KG_report"]
    prompt = kg_report_prompt.format(input_text=describe)
    response = await use_llm_func(prompt)
    logger.info(response)
    
    data = use_string_json_convert_func(response)
    logger.info(data)
    await summary_cache.upsert(data)
    await summary_cache.index_done_callback()
    await llm_response_cache.index_done_callback()
    
    return response


async def query_kg_by_id(protein_node_id: str, go_node_id: str):
    
    await shortest_path(
        protein_node_id=protein_node_id,
        go_node_id=go_node_id
    )

    response = await generate_kg_report(
        protein_node_id=protein_node_id,
        go_node_id=go_node_id
    )
    
    use_model_func = global_config["best_model_func"]
    
    query = "Does this PROTEIN belong to this GO class?"
    sys_prompt_temp = PROMPTS["proteinkg_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            context_data=response
        ),
    )
    logger.info(response)
    

async def query_kg_by_seq(protein_seq: str, go_class: str):
    
    with open(
        osp.join(
            global_config["working_dir"], 
            "protein_seq2id.json"
        ), "r"
    ) as f:
        protein_seq2id = json.load(f)
    with open(
        osp.join(
            global_config["working_dir"],
            "go_name2id.json"
        ), "r"
    ) as f:
        go_name2id = json.load(f)
    if protein_seq in protein_seq2id:
        protein_node_id = f"PROTEIN_{protein_seq2id[protein_seq][0]}"
    else:
        logger.info("Protein sequence is not in the graph ")
        logger.info("Searching in the protein seqdb ")
        results = await search_seqdb(protein_seq)
        # Update the protein node id with top-1 result
        protein_node_id, similarity = results[0]["id"], results[0]["distance"]
        logger.info(
            f"Updated protein node {protein_node_id} with "
            f"highest similarity {similarity} "
        )
    go_node_id = f"GO_{go_name2id[go_class]}"
    
    await shortest_path(
        protein_node_id=protein_node_id,
        go_node_id=go_node_id
    )

    response = await generate_kg_report(
        protein_node_id=protein_node_id,
        go_node_id=go_node_id
    )
    
    use_model_func = global_config["best_model_func"]
    
    query = "Does this PROTEIN belong to this GO class?"
    sys_prompt_temp = PROMPTS["proteinkg_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            context_data=response
        ),
    )
    logger.info(response)
    

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_kg_data", action="store_true")
    parser.add_argument("--generate_seqdb_embedding", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config: dict = json.load(f)
    
    # Update the config with the command line arguments
    config.update(**args.__dict__)
    logger.info(config)
    return config

if __name__ == "__main__":
    
    config = parse_config()
    if config.get("prepare_kg_data", False):
        asyncio.run(upsert_protein_kg())
    if config.get("generate_seqdb_embedding", False):
        asyncio.run(generate_seqdb_embedding())
    
    if config.get("protein_node_id", None) is not None \
        and config.get("go_node_id", None) is not None:
        protein_node_id = f"PROTEIN_{config['protein_node_id']}"
        go_node_id = f"GO_{config['go_node_id']}"
        asyncio.run(
            query_kg_by_id(
                protein_node_id=protein_node_id,
                go_node_id=go_node_id
            )
        )
    elif config.get("protein_seq", None) is not None \
        and config.get("go_class", None) is not None:
        asyncio.run(
            query_kg_by_seq(
                protein_seq=config["protein_seq"],
                go_class=config["go_class"]
            )
        )
    else:
        raise ValueError(
            "Either protein_node_id and go_node_id "
            "or protein_seq and go_class must be provided"
        )

