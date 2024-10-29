from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum

class QueryType(Enum):
    TEXT = "text"
    IMAGE = "image"
    # other types

@dataclass
class QueryItem:
    type: QueryType
    content: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryData:
    items: List[QueryItem]
    
@dataclass
class SearchResultItem:
    query_item: QueryItem
    score: float
    
@dataclass
class SearchResultData:
    results: List[SearchResultItem]

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

search_results = SearchResultData(results=[
    SearchResultItem(
        query_item=query.items[0],
        score=0.95
    ),
    SearchResultItem(
        query_item=query.items[1],
        score=0.85
    )
])