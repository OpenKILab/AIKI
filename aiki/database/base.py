from aiki.multimodal import BaseModalityData

from aiki.multimodal.types import Vector

from abc import ABC, abstractmethod
from bson import ObjectId
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional


class BaseDatabase(ABC):
    def __init__(self):
        super().__init__()


class BaseKVDatabase(BaseDatabase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mget(self, ids: List[ObjectId]):
        pass

    @abstractmethod
    def mset(self, data_list: List[BaseModalityData]):
        pass

    @abstractmethod
    def mdelete(self, ids: List[ObjectId]):
        pass

class BaseVectorDatabase(BaseDatabase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mset(self, data_list: List[BaseModalityData]):
        pass

    @abstractmethod
    def mget(self, ids: List[ObjectId]):
        pass

    @abstractmethod
    def mdelete(self, ids: List[ObjectId]):
        pass

    @abstractmethod
    def query(self, query_embedding: List[Vector], top_k) -> List[ObjectId]:
        pass
    
class BaseRelDatabase(BaseDatabase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mget(self, ids: List[ObjectId]):
        pass

    @abstractmethod
    def mset(self, data_list: List[BaseModalityData]):
        pass

    @abstractmethod
    def mdelete(self, ids: List[ObjectId]):
        pass