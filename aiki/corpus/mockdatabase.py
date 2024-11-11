from abc import ABC, abstractmethod
import json
import os
import chromadb
from typing import List, Literal, Sequence, TypeAlias, TypeVar, TypedDict, Union
from bson import ObjectId
from datetime import datetime
import numpy as np
from numpy._typing import NDArray

ModalityType = Literal["image", "video", "text", "audio"]

T = TypeVar("T")
OneOrMany = Union[T, List[T]]
# vector
PyVector = Union[Sequence[float], Sequence[int]]
Vector: TypeAlias = NDArray[Union[np.int32, np.float32]]
# embedding
PyEmbedding = PyVector
PyEmbeddings = List[PyEmbedding]
Embedding = Vector
Embeddings = List[Embedding]


class KVSchema(TypedDict):
    _id: ObjectId # type: ignore
    modality: ModalityType
    summary: str
    source_encoded_data: str  # Base64 encoded data
    inserted_timestamp: datetime
    parent: List[ObjectId]
    children: List[ObjectId]
    tensor: List[bool]
    
class VectorSchema(TypedDict):
    _id: ObjectId
    embeddings: Union[
                OneOrMany[Embedding],
                OneOrMany[PyEmbedding],
            ]
    doc: str

class NodeSchema(TypedDict):
    ...

class EdgeSchema(TypedDict):
    ...

class DatabaseConnection(ABC):
    @abstractmethod
    def create(self, data):
        pass

    @abstractmethod
    def read(self, id):
        pass

    @abstractmethod
    def update(self, data):
        pass

    @abstractmethod
    def delete(self, id):
        pass

class DatabaseConnectionFactory:
    @staticmethod
    def create_connection(db_type, **kwargs):
        if db_type == 'json_file':
            return JSONFileConnection(**kwargs)
        elif db_type == 'mongodb':
            return MongoDBConnection(**kwargs)
        elif db_type == 'faiss':
            return FAISSConnection(**kwargs)
        elif db_type == 'in_memory':
            return InMemoryConnection(**kwargs)
        elif db_type == 'chroma':
            return ChromaConnection(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

class JSONFileConnection(DatabaseConnection):
    def __init__(self, file_name):
        base_path = os.getcwd()
        path = os.path.join(base_path, "aiki", "corpus", "db", file_name)
        self.file_path = path
        print(self.file_path)
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
        else:
            self.data = []

    def _save_data(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file, default=str)  # Use default=str to handle non-serializable objects like ObjectId

    def create(self, data: KVSchema):
        if any(item["_id"] == data["_id"] for item in self.data):
            raise ValueError(f"ID {data['_id']} already exists.")
        self.data.append(data)
        self._save_data()

    def read(self, id):
        for item in self.data:
            if item["_id"] == id:
                return item
        return None

    def update(self, data: KVSchema):
        for index, item in enumerate(self.data):
            if item["_id"] == data["_id"]:
                self.data[index] = data
                self._save_data()
                return
        raise ValueError(f"ID {data['_id']} does not exist.")

    def delete(self, id):
        for index, item in enumerate(self.data):
            if item["_id"] == id:
                del self.data[index]
                self._save_data()
                return
        raise ValueError(f"ID {id} does not exist.")
        
class MongoDBConnection(DatabaseConnection):
    def __init__(self, uri, db_name, collection_name):
        ...
    ...
        
class FAISSConnection(DatabaseConnection):
    def __init__(self, index_file):
        ...
    ...
    
class InMemoryConnection(DatabaseConnection):
    def __init__(self):
        self.storage = {}

    def create(self, data: KVSchema):
        data_id = data["_id"]  # Ensure to use the correct key for ID
        if data_id in self.storage:
            raise ValueError(f"ID {data_id} already exists.")
        self.storage[data_id] = data

    def read(self, id):
        return self.storage.get(id, None)

    def update(self, data: KVSchema):
        data_id = data["_id"]  # Ensure to use the correct key for ID
        if data_id not in self.storage:
            raise ValueError(f"ID {data_id} does not exist.")
        self.storage[data_id] = data

    def delete(self, id):
        if id in self.storage:
            del self.storage[id]
        else:
            raise ValueError(f"ID {id} does not exist.")

class ChromaConnection(DatabaseConnection):
    def __init__(self, index_file):
        base_path = os.getcwd()
        path = os.path.join(base_path, "aiki", "corpus", "db", index_file)
        print(f"chroma {path}")
        self.chroma_client = chromadb.PersistentClient(path=path) 
        self.collection = self.chroma_client.get_or_create_collection(name=index_file)

    def create(self, data: VectorSchema):
        data_id = str(data["_id"])
        self.collection.upsert(documents=[data["data"]], ids=[data_id])

    def read(self, id):
        results = self.collection.query(query_texts=[id], n_results=1)
        return results

    def update(self, data: VectorSchema):
        data_id = str(data["_id"])  # Ensure to use the correct key for ID
        self.collection.upsert(documents=[data["data"]], ids=[data_id])

    def delete(self, id):
        self.collection.delete(ids=[id])
        
    def query(self, query_texts:List[str], n_results:int) -> chromadb.QueryResult:
        return self.collection.query(query_texts=query_texts, n_results=n_results)
        
        
    def upsert(self, docs:List[str], ids:List[str]):
        self.collection.upsert(
            documents=docs,
            ids=ids
        )
