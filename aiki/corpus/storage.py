from aiki.corpus.database import DatabaseConnection, DatabaseConnectionFactory
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal
from bson import ObjectId
from datetime import datetime

ModalityType = Literal["image", "video", "text", "audio"]

class KVSchema(TypedDict):
    _id: ObjectId
    modality: ModalityType
    summary: str
    source_encoded_data: str  # Base64 encoded data
    inserted_timestamp: datetime
    parent: List[ObjectId]
    children: List[ObjectId]

class VectorSchema(TypedDict):
    ...

class NodeSchema(TypedDict):
    ...

class EdgeSchema(TypedDict):
    ...

@dataclass
class StorageBase(ABC):
    db_connection: DatabaseConnection

    @abstractmethod
    def create(self, data):
        ...

    @abstractmethod
    def read(self, id):
        ...

    @abstractmethod
    def update(self, id, data):
        ...

    @abstractmethod
    def delete(self, id):
        ...

@dataclass
class KVStorage(StorageBase):
    def create(self, data: KVSchema):
        ...

    def read(self, id: str) -> Union[KVSchema, None]:
        ...

    def update(self, data: KVSchema):
        ...

    def delete(self, identifier: str):
        ...

@dataclass
class VectorStorage(StorageBase):
    embedding_func: callable

    def create(self, data: VectorSchema):
        ...

    def read(self, id: str) -> Union[VectorSchema, None]:
        ...

    def update(self, data: VectorSchema):
        ...

    def delete(self, id: str):
        ...

@dataclass
class GraphStorage(StorageBase):
    def create(self, data: Union[NodeSchema, EdgeSchema]):
        ...

    def read(self, id: str) -> Union[NodeSchema, EdgeSchema, None]:
        ...

    def update(self, data: Union[NodeSchema, EdgeSchema]):
        ...

    def delete(self, id: str):
        ...

# Example usage
if __name__ == "__main__":
    # Create a JSON file connection
    db_connection = DatabaseConnectionFactory.create_connection('json_file', file_path='data.json')

    # Initialize storages
    kv_storage = KVStorage(db_connection)
    vector_storage = VectorStorage(db_connection, embedding_func=lambda x: x)
    graph_storage = GraphStorage(db_connection)

    kv_data = {
        "_id": ObjectId(),                          # Ensure this is a valid ObjectId
        "modality": "text",                         # Example modality
        "summary": "This is a summary",
        "source_encoded_data": "SGVsbG8gd29ybGQ=",  # Example Base64 encoded data
        "inserted_timestamp": datetime.now(),
        "parent": [],
        "children": []
    }
    kv_storage.create(kv_data)
