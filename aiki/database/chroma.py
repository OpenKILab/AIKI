from aiki.database import BaseKVDatabase
from aiki.database.base import BaseVectorDatabase
from aiki.multimodal import VectorModalityData, BaseModalityData, Vector
from aiki.serialization import JsonEncoder, JsonDecoder
from aiki.multimodal import ModalityType
import os
import json
from typing import List, Optional, Literal

from bson import ObjectId

class ChromaDB(BaseVectorDatabase):

    def __init__(self,
                 collection_name: str = "default",
                 persist_directory: Optional[str] = None,
                 distance: Literal["l2", "ip", "cosine"] = "cosine",
                 ):
        super().__init__()
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        if persist_directory:
            _client_settings = chromadb.config.Settings(is_persistent=True)
            _client_settings.persist_directory = persist_directory
        else:
            _client_settings = chromadb.config.Settings()
        self._client = chromadb.Client(_client_settings)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": distance}
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
        self._collection.add(
            ids=[str(data._id) for data in data_list],
            embeddings=[data.content for data in data_list],
            metadatas=[data.metadata for data in data_list]
        )

    def mdelete(self, ids: List[ObjectId]):
        self._collection.delete(ids=[str(_id) for _id in ids])

    def query(self, query_embeddings: List[Vector], top_k) -> List[ObjectId]:
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )
        return [[ObjectId(_id) for _id in ids] for ids in result["ids"]]
