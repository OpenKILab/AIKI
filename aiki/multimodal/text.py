from dataclasses import dataclass
from aiki.multimodal.base import BaseModalityData, BaseModalityHandler, ModalityType, BaseModalityHandlerOP
# from aiki.database import BaseKVDatabase

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId

class TextHandlerOP(BaseModalityHandlerOP):
    MSET = "mset"
    MGET = "mget"
    MDELETE = "mdelete"

@dataclass()
class TextModalityData(BaseModalityData):
    modality: ModalityType = ModalityType.TEXT
    content: str = ""

class TextHandler(BaseModalityHandler):

    def mget(self, ids: List[ObjectId]) -> List[TextModalityData]:
        return self.database.mget(ids)

    def mset(self, data_list: List[TextModalityData]):
        self.database.mset(data_list)

    def mdelete(self, ids):
        self.database.mdelete(ids)