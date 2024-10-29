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

# Define a generic base class for storage
@dataclass
class StorageBase(ABC):
    db_connection: DatabaseConnection

    @abstractmethod
    def create(self, id, data):
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

T = TypeVar("T")

@dataclass
class KVStorage(StorageBase, Generic[T]):
    def create(self, identifier: str, data: KVSchema):
        ...

    def read(self, identifier: str) -> Union[KVSchema, None]:
        ...

    def update(self, identifier: str, data: KVSchema):
        ...

    def delete(self, identifier: str):
        ...

@dataclass
class VectorStorage(StorageBase):
    embedding_func: callable

    def create(self, identifier: str, data: VectorSchema):
        ...

    def read(self, identifier: str) -> Union[VectorSchema, None]:
        ...

    def update(self, identifier: str, data: VectorSchema):
        ...

    def delete(self, identifier: str):
        ...

@dataclass
class GraphStorage(StorageBase):
    def create(self, identifier: str, data: Union[NodeSchema, EdgeSchema]):
        ...

    def read(self, identifier: str) -> Union[NodeSchema, EdgeSchema, None]:
        ...

    def update(self, identifier: str, data: Union[NodeSchema, EdgeSchema]):
        ...

    def delete(self, identifier: str):
        ...

# Example usage
if __name__ == "__main__":
    # Create a JSON file connection
    db_connection = DatabaseConnectionFactory.create_connection('json_file', file_path='data.json')

    # Initialize storages
    kv_storage = KVStorage(db_connection)
    vector_storage = VectorStorage(db_connection, embedding_func=lambda x: x)
    graph_storage = GraphStorage(db_connection)

    # Example operations with schemas
    kv_data = {"id": "kv1", "data": {"key1": "value1", "key2": "value2"}}
    kv_storage.create(kv_data["id"], kv_data)
