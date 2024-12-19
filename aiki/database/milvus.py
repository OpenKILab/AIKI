import logging
import sys

import numpy as np
from aiki.database import BaseKVDatabase
from aiki.database.base import BaseVectorDatabase
from aiki.multimodal import VectorModalityData, BaseModalityData
from aiki.multimodal.types import Vector
from aiki.serialization import JsonEncoder, JsonDecoder
from aiki.multimodal import ModalityType
import os
import json
import time
from typing import List, Optional, Literal, Tuple

from bson import ObjectId

class MilvusDB(BaseVectorDatabase):
    def __init__(self,
                    collection_name: str = "default",
                    persist_directory: Optional[str] = None,
                    distance: Literal["l2", "ip", "cosine"] = "cosine",
                    ):
        super().__init__()
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        # default distance COSINE
        self.collection_name = collection_name
        self._client = MilvusClient(persist_directory)
        if self._client.has_collection(collection_name=collection_name):
            self._client.drop_collection(collection_name=collection_name)
        self._client.create_collection(
            collection_name=collection_name,
            dimension=512,
        )


    def mget(self, ids: List[ObjectId]) -> List[VectorModalityData]:
        result_list = []
        for _id in ids:
            result = self._collection.get(ids=[str(_id)],
                                        include=["embeddings", "metadatas"])
            metadatas = result.get("metadatas")
            embeddings = result.get("embeddings")
            result_list.append(
                VectorModalityData(
                    _id=_id,
                    content=embeddings,
                    metadata=metadatas
                )
            )
        return result_list

    def mset(self, data_list: List[VectorModalityData]):
        '''
        self._collection.add(
            ids=[str(data._id) for data in data_list],
            embeddings=[data.content for data in data_list],
            metadatas=[data.metadata for data in data_list]
        )
        '''
        start_time = time.time()
        embeddings = [data.content.tolist() for data in data_list]
        data = [
            {
                "id": str(data._id),
                "vector": embeddings[i],
                "metadatas": data.metadata,
            }
            for i, data in enumerate(data_list)
        ]
        # print(embeddings)
        self._client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        end_time = time.time()  # 记录结束时间
        logging.info(f"Chroma mset operation took {end_time - start_time} seconds")

    def mdelete(self, ids: List[ObjectId]):
        self._collection.delete(ids=[str(_id) for _id in ids])

    def query(self, query_embeddings: List[Vector], top_k, **kwargs) -> List[List[Tuple[ObjectId, ModalityType]]]:
        if "start_time" not in kwargs:
            where = None
        else:
            where = {
                "$and": [
                    {
                    "timestamp": {
                            "$gte": kwargs["start_time"],
                        },
                    },
                    {
                    "timestamp": {
                            "$lte": kwargs["end_time"],
                        },
                    }
                ]
        }
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )
        query_results = []
        for ids, metadatas in zip(result["ids"], result["metadatas"]):
            query_result = []
            for _id, metadata in zip(ids, metadatas):
                modality = ModalityType(metadata["__modality"])
                query_result.append((ObjectId(_id), modality))
            query_results.append(query_result)
        return query_results

# if __name__ == "__main__":
#     from pymilvus import MilvusClient

#     client = MilvusClient("./milvus_demo.db")

