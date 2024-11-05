from aiki.multimodal.base import BaseModality, BaseModalityHandler, ModalityType
from aiki.database import BaseKVDatabase

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId


class TextModality(BaseModality):
    def to_dict(self) -> Dict[str, Any]:
        return {
            '_id': str(self._id),  # 将 ObjectId 转换为字符串
            'modality': self.modality.value,
            'text': self.text,
            'metadata': self.metadata,
        }

    def __init__(self,
                 _id: Union[ObjectId, str],
                 modality: ModalityType,
                 text: str,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(_id, modality, metadata)
        self.text = text


class TextHandler(BaseModalityHandler):
    def __init__(self, database: BaseKVDatabase):
        super().__init__(database)


    def mget(self, ids: List[ObjectId]) -> List[TextModality]:
        return self.database.mget(ids)

    def mset(self, data_list: List[BaseModality]):
        self.database.mset(data_list)

    def mdelete(self, ids):
        self.database.mdelete(ids)