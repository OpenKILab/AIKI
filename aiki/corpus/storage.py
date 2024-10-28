from aiki.corpus.strategies import (
    StorageStrategy,
    TextModalDataStrategy,
    ImageModalDataStrategy,
    VideoModalDataStrategy,
    AudioModalDataStrategy,
    VectorDataStrategy,
    KnowledgeGraphStrategy
)
from aiki.corpus.database import DatabaseConnection, DatabaseConnectionFactory

class BaseStorage:
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.strategies = {}

    def register_strategy(self, data_type: str, strategy: StorageStrategy):
        self.strategies[data_type] = strategy

    def create(self, data_type: str, identifier, data):
        if data_type in self.strategies:
            self.strategies[data_type].create(self.db_connection, identifier, data)
        else:
            raise ValueError(f"No strategy registered for data type: {data_type}")

    def read(self, data_type: str, identifier):
        if data_type in self.strategies:
            return self.strategies[data_type].read(self.db_connection, identifier)
        else:
            raise ValueError(f"No strategy registered for data type: {data_type}")

    def update(self, data_type: str, identifier, data):
        if data_type in self.strategies:
            self.strategies[data_type].update(self.db_connection, identifier, data)
        else:
            raise ValueError(f"No strategy registered for data type: {data_type}")

    def delete(self, data_type: str, identifier):
        if data_type in self.strategies:
            self.strategies[data_type].delete(self.db_connection, identifier)
        else:
            raise ValueError(f"No strategy registered for data type: {data_type}")

# Example usage
if __name__ == "__main__":
    # Create a JSON file connection
    db_connection = DatabaseConnectionFactory.create_connection('json_file', file_path='data.json')

    # Initialize storage
    storage = BaseStorage(db_connection)

    # Register strategies
    storage.register_strategy('text', TextModalDataStrategy())
    storage.register_strategy('image', ImageModalDataStrategy())
    storage.register_strategy('video', VideoModalDataStrategy())
    storage.register_strategy('audio', AudioModalDataStrategy())
    storage.register_strategy('vector', VectorDataStrategy())
    storage.register_strategy('knowledge_graph', KnowledgeGraphStrategy())

    # text data
    storage.create('text', 'doc1', {'content': 'This is a text document.'})
    print(storage.read('text', 'doc1'))
    storage.update('text', 'doc1', {'content': 'This is an updated text document.'})
    print(storage.read('text', 'doc1'))
    storage.delete('text', 'doc1')
    print(storage.read('text', 'doc1'))
    
    # image data
    print("Image Data Operations:")
    storage.create('image', 'img1', {'content': 'This is image data.'})
    print("Read Image:", storage.read('image', 'img1'))
    storage.update('image', 'img1', {'content': 'This is updated image data.'})
    print("Updated Image:", storage.read('image', 'img1'))
    storage.delete('image', 'img1')
    print("Deleted Image:", storage.read('image', 'img1'))

    # vector data
    print("\nVector Data Operations:")
    storage.create('vector', 'vec1', [0.1, 0.2, 0.3])
    print("Read Vector:", storage.read('vector', 'vec1'))
    storage.update('vector', 'vec1', [0.4, 0.5, 0.6])
    print("Updated Vector:", storage.read('vector', 'vec1'))
    storage.delete('vector', 'vec1')
    print("Deleted Vector:", storage.read('vector', 'vec1'))