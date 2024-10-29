from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

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