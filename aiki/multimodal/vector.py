from dataclasses import dataclass
from aiki.multimodal.base import BaseModalityData, BaseModalityHandler, BaseModalityHandlerOP, ModalityType
# from aiki.database import BaseVectorDatabase

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional, Callable
from bson import ObjectId

from aiki.multimodal.types import Vector

class VectorHandlerOP(BaseModalityHandlerOP):
    MSET = "mset"
    MGET = "mget"
    MDELETE = "mdelete"
    UPSERT = "upsert"
    QUERY = "query"

@dataclass()
class VectorModalityData(BaseModalityData):
    modality: ModalityType = ModalityType.VECTOR
    content: Vector = None

class VectorHandler(BaseModalityHandler):
    def __init__(self, database: "BaseVectorDatabase", embedding_func: Optional[Callable[[Any], Any]] = None):
        super().__init__(database)
        self.embedding_func = embedding_func

    def mget(self, ids: List[ObjectId]) -> List[VectorModalityData]:
        return self.database.mget(ids)

    def mset(self, data_list: List[VectorModalityData]):
        self.database.mset(data_list)

    def mdelete(self, ids: List[ObjectId]):
        self.database.mdelete(ids)

    def query(self,
              query_data: Optional[List[Any]] = None,
              query_embeddings: Optional[List[Vector]] = None,
              top_k: int = 10) -> List[ObjectId]:
        if query_embeddings is None:
            if query_data is None:
                raise ValueError("query_embeddings and query_data cannot be both None")
            else:
                query_embeddings = [self.embedding_func([data])[0] for data in query_data]
        return self.database.query(query_embeddings, top_k = top_k)

    def upsert(self, data: List[BaseModalityData]):
        vector_data = []
        for item in data:
            _id = item._id
            content = self.embedding_func([item.content])
            metadata = item.metadata or {}
            metadata.update({"__modality": item.modality.value})
            vector_data.append(
                VectorModalityData(_id=_id, content=content, metadata=metadata)
            )
        self.mset(vector_data)