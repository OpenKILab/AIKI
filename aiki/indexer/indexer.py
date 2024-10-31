from aiki.corpus.database import DatabaseConnectionFactory
from aiki.corpus.storage import BaseStorage, KVStorage
from bson import ObjectId
from datetime import datetime
from abc import ABC, abstractmethod

from aiki.indexer.chunker import BaseChunker, FixedSizeChunker
from aiki.modal.retrieval_data import RetrievalData, RetrievalType

class BaseIndexer(ABC):
    def __init__(self, model_path, sourcedb: BaseStorage, vectordb: BaseStorage, chunker:BaseChunker = FixedSizeChunker()):
        self.model_path = model_path
        self.sourcedb = sourcedb  # Source database storage
        self.vectordb = vectordb  # Vector database storage
        self.chunker = chunker
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(BaseIndexer):
    def index(self, data: RetrievalData):
        for retreval_data in data.items:
            if retreval_data.type != RetrievalType.TEXT:
                raise ValueError(f"Unsupported data type: {retreval_data.type}")
            id = ObjectId()
            dataSchema = {
                "id": id,
                "modality": "text",
                "summary": "",
                "source_encoded_data": retreval_data["content"],
                "inserted_timestamp": datetime.now(),
                "parent": [],
                "children": []
            }
            self.sourcedb.create(dataSchema)
            chunks = self.chunker.chunk(data)
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
        
class MultimodalIndexer(BaseIndexer):
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

class KnowledgeGraphIndexer(BaseIndexer):
    ...
    

# 多模态数据生成文本摘要
class BaseSummaryGenerator(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        
    def generate_summary(self, data):
        ...
        
class ModelSummaryGenerator(BaseSummaryGenerator):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = self.load_model(self.model_path) 
    
    def load_model(self, model_path):
        # load tokenizer and model
        ...
        
    def generate_summary(self, data):
        ...

class APISummaryGenerator(BaseSummaryGenerator):
    def __init__(self):
        super().__init__()
        
    def generate_summary(self, data):
        # request with config
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