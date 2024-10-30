from aiki.corpus.database import DatabaseConnectionFactory
from aiki.corpus.storage import BaseStorage, KVStorage
from bson import ObjectId
from datetime import datetime

from aiki.modal.retrieval_data import RetrievalData, RetrievalType

class Indexer:
    def __init__(self, model_path, sourcedb: BaseStorage, vectordb: BaseStorage):
        self.model_path = model_path
        self.sourcedb = sourcedb  # Source database storage
        self.vectordb = vectordb  # Vector database storage
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(Indexer):
    def index(self, data: RetrievalData):
        for retreval_data in data.items:
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
            chunks = self.chunker.chunk(data)
            # 将chunk 插入到 sourcedb中
            for data in chunks:
                cur_id = ObjectId()
                dataSchema = {
                    "id": cur_id,
                    "modality": "text",
                    "summary": "",
                    "source_encoded_data": data,
                    "inserted_timestamp": datetime.now(),
                    "parent": [id],
                    "children": []
                }
                self.sourcedb.create(dataSchema)
                '''
                VectorSchema = {
                    "id": cur_id,
                    "data": data
                }
                self.vectordb.create(VectorSchema)
                '''
        
class MultimodalIndexer(Indexer):
    def __init__(self, model_path, sourcedb: BaseStorage, vectordb: BaseStorage):
        super().__init__(model_path, sourcedb, vectordb)
        self.text_indexer = TextIndexer(model_path, sourcedb, vectordb)
    
    def index(self, data):
        if data.type == RetrievalType.TEXT:
            self.text_indexer.index(data)
        elif data.type == RetrievalType.IMAGE:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data.type}")

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