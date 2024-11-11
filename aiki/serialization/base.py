import importlib
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional, Type
from bson import ObjectId
from enum import Enum
from datetime import datetime
import json
from dataclasses import dataclass

from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs import modules

SERIALIZABLE_LIST = {
    "ModalityType": ("aiki", "multimodal", "ModalityType"),
    "BaseModalityData": ("aiki", "multimodal", "BaseModalityData"),
    "TextModalityData": ("aiki", "multimodal", "TextModalityData"),
    "ImageModalityData": ("aiki", "multimodal", "ImageModalityData"),
}

SerializableValue = Union[
    str,
    int,
    float,
    bool,
    None,
    datetime,
    ObjectId,
    Enum,
    'Serializable',
    List['SerializableValue'],
    Dict[str, 'SerializableValue']
]

T = TypeVar('T', bound='JSONSerializable')

class JsonEncoder(json.JSONEncoder):

    """自定义JSON编码器"""
    def default(self, obj: Any) -> Any:

        if isinstance(obj, Serializable):
            return {
                "__type__": type(obj).__name__,
                "__data__": obj.to_dict()
            }
        elif isinstance(obj, datetime):
            return {
                "__type__": "datetime",
                "__data__": obj.isoformat()
            }
        elif isinstance(obj, ObjectId):
            return {
                "__type__": "ObjectId",
                "__data__": str(obj)
            }
        elif isinstance(obj, Enum):
            return {
                "__type__": "Enum",
                "__enum__": type(obj).__name__,
                "__data__": obj.value
            }
        return super().default(obj)

def get_cls(cls_name: str) -> Type[T]:
    """Get a class by name"""
    return getattr(importlib.import_module('.'.join(SERIALIZABLE_LIST[cls_name][:-1])), cls_name)

class JsonDecoder(json.JSONDecoder):
    """Custom JSON decoder to handle special types like datetime and ObjectId"""

    def __init__(self, *args, **kwargs):
        # Call the super with a custom object hook for decoding
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, obj: Dict[str, Any]) -> Any:
        """Hook to process each JSON object during decoding"""
        if "__type__" in obj:
            type_name = obj["__type__"]
            data = obj["__data__"]

            if type_name == "datetime":
                return datetime.fromisoformat(data)
            elif type_name == "ObjectId":
                return ObjectId(data)
            elif type_name == "Enum":
                enum_name = obj["__enum__"]
                enum_value = obj["__data__"]
                # Look up the Enum class by name
                enum_class = get_cls(enum_name)
                if issubclass(enum_class, Enum):
                    return enum_class(enum_value)

            # Attempt to get and instantiate custom Serializable classes
            try:
                cls = get_cls(type_name)
                if cls and issubclass(cls, Serializable):
                    return cls.from_dict(data)
            except (KeyError, TypeError):
                pass  # Return the object unchanged if we can't deserialize it

        return obj  # Return as is if no special handling is required

@dataclass
class Serializable:
    """可序列化对象的基类"""

    def to_dict(self) -> Dict[str, SerializableValue]:
        """将对象转换为可序列化的字典"""
        return {
            k: self._serialize_value(v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, SerializableValue]) -> T:
        """从字典创建对象"""
        return cls(**{
            k: cls._deserialize_value(v)
            for k, v in data.items()
        })

    @staticmethod
    def _serialize_value(value: Any) -> SerializableValue:
        """序列化单个值"""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        elif isinstance(value, Serializable):
            return value.to_dict()
        elif isinstance(value, list):
            return [Serializable._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: Serializable._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, datetime):
            return {
                "__type__": "datetime",
                "__data__": value.isoformat()
            }
        elif isinstance(value, ObjectId):
            return {
                "__type__": "ObjectId",
                "__data__": str(value)
            }
        elif isinstance(value, Enum):
            return {
                "__type__": "Enum",
                "__enum__": value.__class__.__name__,
                "__data__": value.value
            }
        raise TypeError(f"无法序列化类型: {type(value)}")

    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """反序列化单个值"""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        elif isinstance(value, list):
            return [Serializable._deserialize_value(item) for item in value]
        elif isinstance(value, dict):
            if "__type__" in value:
                type_name = value["__type__"]
                data = value["__data__"]

                if type_name == "datetime":
                    return datetime.fromisoformat(data)
                elif type_name == "ObjectId":
                    return ObjectId(data)
                elif type_name == "Enum":
                    enum_name = value["__enum__"]
                    # Look up the Enum class by name
                    enum_class = get_cls(enum_name)
                    if issubclass(enum_class, Enum):
                        return enum_class(data)
                # 尝试获取并实例化自定义类
                try:
                    cls = get_cls(type_name)
                    if issubclass(cls, Serializable):
                        return cls.from_dict(data)
                except (KeyError, TypeError):
                    pass

            return {k: Serializable._deserialize_value(v) for k, v in value.items()}
        return value

    def to_json(self, indent: Optional[int] = None) -> str:
        """将对象转换为JSON字符串"""
        return json.dumps(self, cls=JsonEncoder, indent=indent)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        if isinstance(data, dict) and "__type__" in data:
            return cls.from_dict(data["__data__"])
        raise ValueError("Invalid JSON format")