from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

class QueryType(Enum):
    TEXT = "text"
    IMAGE = "image"
    # Add more types as needed

@dataclass
class QueryItem:
    type: QueryType
    content: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryData:
    items: List[QueryItem]

# Example usage
query = QueryData(items=[
    QueryItem(
        type=QueryType.TEXT,
        content={
            "value": "How does AI work? Explain it in simple terms.",
            "annotations": []
        }
    ),
    QueryItem(
        type=QueryType.IMAGE,
        content={
            "value": "base64_encoded_data",
            "annotations": []
        }
    )
])