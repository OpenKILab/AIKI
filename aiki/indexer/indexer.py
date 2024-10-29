from aiki.corpus.database import DatabaseConnectionFactory
from aiki.corpus.storage import BaseStorage, KVStorage
from bson import ObjectId
from datetime import datetime

class Indexer:
    def __init__(self, model_path, sourcedb: BaseStorage, vectordb: BaseStorage):
        self.model_path = model_path
        self.sourcedb = sourcedb  # Source database storage
        self.vectordb = vectordb  # Vector database storage
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(Indexer):
    def index(self, data):
        id = ObjectId()
        dataSchema = {
            "id": id,
            "modality": "text",
            "summary": "",
            "source_encoded_data": data,
            "inserted_timestamp": datetime.now(),
            "parent": [],
            "children": []
        }
        self.sourcedb.create(dataSchema)
        self.vectordb.create(dataSchema)

class MultimodalIndexer(Indexer):
    ...

class KnowledgeGraphIndexer(Indexer):
    ...

# Example usage
if __name__ == "__main__":
    # JSON file
    source_db_connection = DatabaseConnectionFactory.create_connection('json_file', file_path='data.json')
    
    faiss_connection = DatabaseConnectionFactory.create_connection('faiss', index_file='index.faiss')

    # Initialize source and vector databases
    sourcedb = BaseStorage(source_db_connection)
    vectordb = BaseStorage(faiss_connection)

    # Initialize specific indexers
    text_indexer = TextIndexer(model_path='path/to/model', sourcedb=sourcedb, vectordb=vectordb)

    # Example data to index
    data = "Example text data"

    # Index the data
    text_indexer.index(data)