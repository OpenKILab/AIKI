from abc import ABC, abstractmethod
from aiki.corpus.database import DatabaseConnection

class StorageStrategy(ABC):
    @abstractmethod
    def create(self, db_connection: DatabaseConnection, id, data):
        pass

    @abstractmethod
    def read(self, db_connection: DatabaseConnection, id):
        pass

    @abstractmethod
    def update(self, db_connection: DatabaseConnection, id, data):
        pass

    @abstractmethod
    def delete(self, db_connection: DatabaseConnection, id):
        pass

class TextModalDataStrategy(StorageStrategy):
    ...

class ImageModalDataStrategy(StorageStrategy):
    ...

class VideoModalDataStrategy(StorageStrategy):
    ...

class AudioModalDataStrategy(StorageStrategy):
    ...
    
class VectorDataStrategy(StorageStrategy):
    ...
    
class KnowledgeGraphStrategy(StorageStrategy):
    ...