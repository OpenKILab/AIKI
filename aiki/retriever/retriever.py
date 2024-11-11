from abc import ABC, abstractmethod
from typing import List, Dict, Protocol
from aiki.corpus.mockdatabase import DatabaseConnectionFactory
from aiki.database.base import BaseKVDatabase, BaseVectorDatabase
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.modal.retrieval_data import RetrievalData, RetrievalType, RetrievalItem
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.text import TextHandler
from aiki.multimodal.vector import VectorHandler, VectorHandlerOP
from chromadb.utils import embedding_functions

class DataPool:
    def __init__(self):
        self.pool = {}

    def add(self, component_name: str, data: RetrievalData):
        self.pool[component_name] = data
    
    def get(self, component_name: str) -> RetrievalData:
        return self.pool.get(component_name, None)

class PreRetrieverComponent(ABC):
    @abstractmethod
    def process(self, data: RetrievalData) -> RetrievalData:
        ...
        
class PostRetrieverComponent(ABC):
    @abstractmethod
    def process(self, data: RetrievalData) -> RetrievalData:
        ...
        
class RecallStrategy(Protocol):
    def search(self, query: RetrievalData, num: int) -> RetrievalData:
        """Search for relevant data based on the strategy."""
        ...

class BaseRetriever(ABC):
    def __init__(self, config, preretriever_components: List[PreRetrieverComponent] = None, postretriever_components: List[PostRetrieverComponent] = None, recall_strategies: List[RecallStrategy] = None):
        self.topk = config.get("retrieval_topk", 4)
        self.recall_strategies = recall_strategies or []
        self.config = config
        self.database = {}
        self.data_pool = DataPool()
        self.preretriever_components = []
        self.postretriever_components = []
        self.processor = MultiModalProcessor()
        
        self.load_config()
    
    def load_config(self):
        self.database = {db_config['type']: self.create_database(db_config) for db_config in self.config['databases']}
        for db_type, db_instance in self.database.items():
            if isinstance(db_instance, BaseKVDatabase):
                self.processor.register_handler(ModalityType.TEXT, TextHandler(database=db_instance))
            elif isinstance(db_instance, BaseVectorDatabase):
                self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=db_instance, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
    
    
    def create_database(self, db_config):
        db_type = db_config['type']
        db_args = db_config['args']
        
        if db_type == "json_file":
            return JSONFileDB(**db_args)
        elif db_type == "chroma":
            return ChromaDB(**db_args)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def pre_retrieve(self, query: RetrievalData):
        for component in self.preretriever_components:
            query = component.process(query)
            self.data_pool.add(component.__class__.__name__, query)

    def post_retrieve(self, query: RetrievalData, results: RetrievalData):
        for component in self.postretriever_components:
            results = component.process(results)
            self.data_pool.add(component.__class__.__name__, results)

    def _search(self, num: int = None):
        ...

    def search(self, query: RetrievalData, num: int = None) -> List[RetrievalData]:
        self.pre_retrieve(query)
        self._search(num)
        self.post_retrieve()
        results = []
        for component in self.postretriever_components:
            processed_data = component.process(query)
            self.data_pool.add(component.__class__.__name__, processed_data)
            results.append(processed_data)
        # TODO: 对List[RetrievalData]去重和打包
        return results

    
class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
    def _search(self, query: RetrievalData, num: int = 4):
        queries = [q.content for q in query.items if q.type == RetrievalType.TEXT]
        result = self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, queries)

    def search(self, query: RetrievalData, num: int = 4) -> RetrievalData:
        results = self._search(query, num)
        return results
        
if __name__ == "__main__":
    config = {
            "retrieval_topk": 2,
            "databases": [
                {
                    "type": "json_file",
                    "args": {
                        "file_path": "./aiki/corpus/db/data.json",
                    }
                },
                {
                    "type": "chroma",
                    "args": {
                        "collection_name": "text_index",
                        "persist_directory": "./aiki/corpus/db/test_index",
                    }
                }
            ]
        }
    dense_retriever = DenseRetriever(config=config)
    retrieval_data = RetrievalData(items=[
        RetrievalItem(
            content= "Marley",
            type= RetrievalType.TEXT
        )
    ])
    print(dense_retriever.search(retrieval_data))