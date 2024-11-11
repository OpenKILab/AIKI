from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum

from bson import ObjectId

from aiki.multimodal.base import ModalityType
from aiki.serialization.base import Serializable

class RetrievalType(Enum):
    TEXT = "text"
    IMAGE = "image"
    # other types

@dataclass
class RetrievalItem:
    type: RetrievalType
    content: str

@dataclass
class RetrievalData:
    items: List[RetrievalItem]

# Example usage
query = RetrievalData(items=[
    RetrievalItem(
        type=RetrievalType.TEXT,
        content="How does AI work? Explain it in simple terms.",
    ),
    RetrievalItem(
        type=RetrievalType.IMAGE,
        content="base64_encoded_data",
    )
])

@dataclass
class KVSchema(Serializable):
    _id: ObjectId
    modality: ModalityType
    summary: str
    source_encoded_data: str
    inserted_timestamp: datetime
    parent: List[ObjectId] = field(default_factory=list)
    children: List[ObjectId] = field(default_factory=list)
    tensor: List[bool] = field(default_factory=list)
    