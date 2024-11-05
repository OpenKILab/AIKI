from aiki.multimodal import BaseModality

from abc import ABC, abstractmethod
from bson import ObjectId
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional


class BaseDatabase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def close(self):
        ...


class BaseKVDatabase(BaseDatabase):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mget(self, ids: List[ObjectId]):
        pass

    @abstractmethod
    def mset(self, data_list: List[BaseModality]):
        pass

    @abstractmethod
    def mdelete(self, ids: List[ObjectId]):
        pass