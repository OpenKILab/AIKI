from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional, Type
from bson import ObjectId
from enum import Enum
from aiki.serialization import SerializableValue, Serializable


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VECTOR = "vector"
    UNKNOWN = "unknown"

@dataclass()
class BaseModalityData(Serializable):
    _id: ObjectId
    modality: ModalityType = ModalityType.UNKNOWN
    content: Any = None
    metadata: Dict[str, SerializableValue] = field(default_factory=dict)

class BaseModalityHandler(ABC):
    def __init__(self, database):
        self.database = database

class BaseModalityHandlerOP(Enum):
    pass

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

    def execute_operation(self, modality_type: ModalityType, operation: BaseModalityHandlerOP, *args, **kwargs):
        """执行特定模态的操作"""
        handler = self._get_handler(modality_type)
        if not hasattr(handler, operation.value):
            raise ValueError(f"Operation {operation} not supported for modality: {modality_type}")
        
        operation_func = getattr(handler, operation.value)
        if asyncio.iscoroutinefunction(operation_func):
            return asyncio.run(operation_func(*args, **kwargs))
        else:
            return operation_func(*args, **kwargs)
    