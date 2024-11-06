from dataclasses import dataclass
from aiki.multimodal.base import BaseModalityData, BaseModalityHandler, ModalityType, BaseModalityHandlerOP, Serializable
from aiki.database import BaseKVDatabase

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId
from enum import Enum

class TextHandlerOP(BaseModalityHandlerOP):
    MSET = "mset"
    MGET = "mget"
    MDELETE = "mdelete"

@dataclass()
class TextModalityData(BaseModalityData):
    text: str = ""

class TextHandler(BaseModalityHandler):
    def __init__(self, database: BaseKVDatabase):
        super().__init__(database)


    def mget(self, ids: List[ObjectId]) -> List[TextModalityData]:
        return self.database.mget(ids)

    def mset(self, data_list: List[BaseModalityData]):
        self.database.mset(data_list)

    def mdelete(self, ids):
        self.database.mdelete(ids)