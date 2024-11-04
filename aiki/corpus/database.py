from abc import ABC, abstractmethod
import os

import json
from pymongo import MongoClient
import chromadb

class DatabaseConnection(ABC):
    @abstractmethod
    def create(self, data):
        pass

    @abstractmethod
    def read(self, _id):
        pass

    @abstractmethod
    def update(self, data):
        pass

    @abstractmethod
    def delete(self, _id):
        pass

class DatabaseConnectionFactory:
    @staticmethod
    def create_connection(db_type, **kwargs):
        if db_type == 'json_file':
            return JSONFileDBConnection(**kwargs)
        elif db_type == 'mongodb':
            return MongoDBConnection(**kwargs)
        elif db_type == 'faiss':
            return FAISSDBConnection(**kwargs)
        elif db_type == 'chroma':
            return ChromaDBConnection(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

class JSONFileDBConnection(DatabaseConnection):
    def __init__(self, file_path):
        self.file_path = file_path
        # Initialize empty JSON file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)

    def _read_file(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

    def _write_file(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def create(self, data):
        _id = data.get('_id')
        file_data = self._read_file()

        if _id in file_data:
            raise ValueError(f"Record with ID {_id} already exists")

        file_data[_id] = data
        self._write_file(file_data)
        return True

    def read(self, _id):
        file_data = self._read_file()

        if _id not in file_data:
            raise ValueError(f"Record with ID {_id} not found")

        return file_data[id]

    def update(self, data):
        _id = data.get('_id')
        file_data = self._read_file()

        if _id not in file_data:
            raise ValueError(f"Record with ID {_id} not found")

        file_data[_id] = data
        self._write_file(file_data)
        return True

    def delete(self, _id):
        file_data = self._read_file()

        if _id not in file_data:
            raise ValueError(f"Record with ID {_id} not found")

        del file_data[_id]
        self._write_file(file_data)

        return True


class MongoDBConnection(DatabaseConnection):
    def __init__(self, uri, db_name, collection_name):
        """Initialize MongoDB connection"""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def create(self, data):
        _id = data.get('_id')

        if self.collection.find_one({"_id": _id}):
            raise ValueError(f"Record with ID {_id} already exists")

        result = self.collection.insert_one(data)
        return bool(result.inserted_id)

    def read(self, _id):
        document = self.collection.find_one({"_id": _id})
        if not document:
            raise ValueError(f"Record with ID {_id} not found")

        return document

    def update(self, data):
        _id = data.get('_id')

        if not self.collection.find_one({"_id": _id}):
            raise ValueError(f"Record with ID {_id} not found")

        result = self.collection.update_one(
            {"_id": _id},
            {"$set": data}
        )
        return bool(result.modified_count)

    def delete(self, _id):
        result = self.collection.delete_one({"_id": _id})
        if result.deleted_count == 0:
            raise ValueError(f"Record with ID {_id} not found")

        return True

    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()

class FAISSDBConnection(DatabaseConnection):
    def __init__(self, index_file):
        ...
    ...


class ChromaDBConnection(DatabaseConnection):
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB connection"""
        self.client = chromadb.Client(chromadb.settings.Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def create(self, data):
        _id = data.get('_id')

        try:
            self.collection.get(ids=[_id])
            raise ValueError(f"Record with ID {_id} already exists")
        except Exception:
            pass

        embedding = data.get('embedding')

        if embedding is None:
            raise ValueError("Embedding is required")

        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            ids=[_id]
        )
        return True

    def read(self, _id) :
        try:
            result = self.collection.get(ids=[_id])
        except Exception:
            raise ValueError(f"Record with ID {_id} not found")

        # Format response
        return {
            'embedding': result['embeddings'][0] if result['embeddings'] else None,
        }

    def update(self, data):
        _id = data.get('_id')
        try:
            self.collection.get(ids=[_id])
        except Exception:
            raise ValueError(f"Record with ID {_id} not found")

        embedding = data.get('embedding')

        if embedding is None:
            raise ValueError("Embedding is required for update")

        # Update in ChromaDB
        self.collection.update(
            embeddings=[embedding],
            ids=[_id]
        )
        return True

    def delete(self, _id) :
        try:
            self.collection.get(ids=[_id])
        except Exception:
            raise ValueError(f"Record with ID {_id} not found")

        self.collection.delete(ids=[_id])
        return True

    def __del__(self):
        if hasattr(self, 'client'):
            self.client.reset()

