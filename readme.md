# Protein-GraphRAG

Based on [GraphRAG](https://github.com/gusye1234/nano-graphrag), a simple, easy-to-hack GraphRAG implementation with ProteinKG.

## Install

**Install from source** (recommend)

```shell
# clone this repo first
cd protein-graphrag
pip install -e .
```

## Quick Start

If you first time to use this tool, you need to download the data and preprocess the protein-go knowledge graph.

```shell
gdown 'https://drive.google.com/uc?id=1iTC2-zbvYZCDhWM_wxRufCvV6vvPk8HR'
# ProteinKG25.zip
unzip ProteinKG25.zip

# Preprocessed KG will take about 500 MB with pickle.
python proteinKG.py --prepare_kg_data

# Generate embedding for protein sequence with ESM C.
python proteinKG.py --generate_seqdb_embedding
```

Search for the relationship between a protein with ID 309507 and a GO term with ID 219.
```shell
# fill in your LLM API key in proteinKG.py first, DeepSeek for example
export LLM_BASE_URL=https://api.deepseek.com
export LLM_API_KEY=sk-4e9ca3862cc44b50852afdc516a1b360
export LLM_MODEL=deepseek-chat

python proteinKG.py \
    --config example/example_id.json
```

example/example_id.json
```json
{
    "protein_node_id": 309507,
    "go_node_id": 219
}
```

Or you can directly use the protein sequence and GO class to search.
```shell
python proteinKG.py \
    --config example/example_seq.json \
```

example/example_seq.json
```json
{
    "protein_seq": "MKTPLTEAIAAADLRGSYLSNTELQAVFGRFNRARAGLEAARAFANNGKKWAEAAANHVYQKFPYTTQMQGPQYASTPEGKAKCVRDIDHYLRTISYCCVVGGTGPLDDYVVAGLKEFNSALGLSPSWYIAALEFVRDNHGLTGDVAGEANTYINYAINALS",
    "go_class": "GO:0000322"
}
```

The LLM will generate the relationship between the protein and the GO term.
```
False. PROTEIN P00309 does not belong to the GO:0000322 class.
```
All intermediate results are stored in a cache dictionary `proteinKG_cache/PROTEIN_309507-GO_219`.

Its structure is as follows:
```
PROTEIN_309507-GO_219
├── entity.csv
├── kv_store_kg_report_llm_response_cache.json
├── kv_store_kg_report_summary_cache.json
└── relation.csv
```

`entity.csv` contains the protein and GO term information.

|entity_attr|entity_name|description|
|---|---|---|
|PROTEIN|P00309|MKTPLTEAIAAADLR...YINYAINALS|
|GO|GO:0016020|membrane [Component]: A lipid bilayer along with all the proteins and protein complexes embedded in it an attached to it.|
|PROTEIN|P60766|MQTIKCVVVGDGAVG...EPKKSRRCVLL|
|GO|GO:0000322|storage vacuole [Component]: A vacuole that functions primarily in the storage of materials, including nutrients, pigments, waste products, and small molecules.|

`relation.csv` contains the relationship between the protein and GO term.

|source|target|description|
|---|---|---|
|P00309|GO:0016020|P00309 (PROTEIN) located_in GO:0016020 (GO)|
|GO:0016020|P60766|GO:0016020 (GO) located_in P60766 (PROTEIN)|
|P60766|GO:0000322|P60766 (PROTEIN) located_in GO:0000322 (GO)|

`kv_store_kg_report_llm_response_cache.json` contains the LLM response.

`kv_store_kg_report_summary_cache.json` contains the summary of the LLM response.

```json
{
  "title": "BELONGING of PROTEIN P00309 to GO:0000322",
  "summary": "The PROTEIN P00309 is located in the membrane GO:0016020. GO:0016020 is located in the PROTEIN P60766, which in turn is located in the storage vacuole GO:0000322. This path suggests a weak relationship between PROTEIN P00309 and the storage vacuole GO:0000322.",
  "rating": 1.0,
  "rating_explanation": "The impact severity rating is low due to the indirect path connecting PROTEIN P00309 to the storage vacuole GO:0000322, with an intermediate PROTEIN P60766."
}
```

