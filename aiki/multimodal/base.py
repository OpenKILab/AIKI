from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId
from enum import Enum

# 定义基础的JSON可序列化类型
JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, List['JsonValue'], Dict[str, 'JsonValue']]

# 或者更具体地包含你的特定类型
SerializableValue = Union[
    str,
    int,
    float,
    bool,
    None,
    datetime,
    ObjectId,
    List['SerializableValue'],
    Dict[str, 'SerializableValue']
]

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VECTOR = "vector"

class BaseModality(ABC):
    def __init__(
        self,
        _id: Union[ObjectId, str],
        modality: ModalityType,
        metadata: Optional[Dict[str, SerializableValue]] = None
    ):
        self._id = _id
        self.modality = modality
        self.metadata = metadata or {}

    @property
    def id(self):
        return self._id

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...

class BaseModalityHandler(ABC):
    def __init__(self, database):
        self.database = database


class MultiModalProcessor:
    def __init__(self):
        self.handlers: Dict[str, BaseModalityHandler] = {}

    def register_handler(self, modality_type: ModalityType, handler: BaseModalityHandler):
        if not isinstance(handler, BaseModalityHandler):
            raise TypeError("Handler must implement ModalityHandler interface")
        self.handlers[modality_type.value] = handler

    def _get_handler(self, modality_type: ModalityType) -> BaseModalityHandler:
        handler = self.handlers.get(modality_type.value)
        if not handler:
            raise ValueError(f"No handler registered for modality: {modality_type}")
        return handler

    def execute_operation(self, modality_type: ModalityType, operation: str, *args, **kwargs):
        """执行特定模态的特殊操作"""
        handler = self._get_handler(modality_type)
        if not hasattr(handler, operation):
            raise ValueError(f"Operation {operation} not supported for modality: {modality_type}")
        return getattr(handler, operation)(*args, **kwargs)
