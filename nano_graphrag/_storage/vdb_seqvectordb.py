import asyncio
import torch
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from esm.models.esmc import ESMC
from tqdm import tqdm

from .._utils import logger


@dataclass
class SeqVectorDBStorage:
    namespace: str
    global_config: dict
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self.model = ESMC.from_pretrained(
            self.global_config["esmc_model"]
        ).to(self.global_config["device"])
        self.model.eval()
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["embedding_dim"], storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def embedding_func(self, sequences: list[str]) -> np.ndarray:
        input_ids = self.model._tokenize(sequences)
        sequence_id = (input_ids == self.model.tokenizer.pad_token_id)
        with (
            torch.no_grad(),
            torch.autocast(
                enabled=True,
                device_type=torch.device(input_ids.device).type,
                dtype=torch.bfloat16,
            ),
        ):
            embeddings = self.model(input_ids).embeddings
        batch_size = embeddings.shape[0]
        embed_list = []
        for i in range(batch_size):
            embed, mask = embeddings[i], ~sequence_id[i]
            unpad_embed = embed[mask][1:-1] # Remove cls and eos
            embed_list.append(
                unpad_embed.mean(dim=0, keepdim=True)
                .detach().to(torch.float32).cpu().numpy()
            )
        return np.concatenate(embed_list)

    async def upsert(self, data: dict[str, str]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                "__seq__": v,
            }
            for k, v in data.items()
        ]
        contents = [seq for seq in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = [
            await self.embedding_func(batch) 
            for batch in tqdm(batches, desc="Embedding")
        ]
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
