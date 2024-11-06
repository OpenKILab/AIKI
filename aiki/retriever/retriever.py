from abc import ABC, abstractmethod
from typing import List, Dict, Protocol
from aiki.corpus.mockdatabase import DatabaseConnectionFactory
from aiki.modal.retrieval_data import RetrievalData, RetrievalType, RetrievalItem

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
        
        self.load_config()
    
    def load_config(self):
        for db_config in self.config.get("databases", []):
            db_type = db_config.get("type")
            db_args = db_config.get("args")
            if db_type:
                connection = DatabaseConnectionFactory.create_connection(db_type=db_type, **db_args)
                self.database[db_type] = connection

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
        
    def _search(self, query: RetrievalData, num: int = 4) -> RetrievalData:
        queries = [q.content for q in query.items if q.type == RetrievalType.TEXT]
        results = self.database['chroma'].query(queries, n_results = num)
        return None
    
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
                        "file_name": "json_file",
                    }
                },
                {
                    "type": "chroma",
                    "args": {
                        "index_file": "chroma_index",
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