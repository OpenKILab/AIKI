from typing import List, Dict, Protocol
from aiki.modal.interaction_data import QueryData, SearchResultData

class RecallStrategy(Protocol):
    def search(self, query: QueryData, num: int) -> List[Dict[str, str]]:
        """Search for relevant data based on the strategy."""
        ...

class BaseRetriever:
    def __init__(self, config, recall_strategies: List[RecallStrategy] = None):
        self.topk = config.get("retrieval_topk", 10)
        self.recall_strategies = recall_strategies or []

    def pre_retrieve(self, query: QueryData) -> QueryData:
        """Pre-process the query before searching."""
        ...

    def post_retrieve(self, results: SearchResultData) -> QueryData:
        """Post-process the results after searching."""
        ...

    def _search(self, query: QueryData, num: int = None) -> SearchResultData:
        ...

    def search(self, query: QueryData, num: int = None) -> QueryData:
        """Retrieve topk relevant data in corpus."""
        query = self.pre_retrieve(query)
        results = self._search(query, num)
        results = self.post_retrieve(results)
        return results
    
    
class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.storages = self._initialize_storages(config)

    def _initialize_storages(self, config):
        '''
        {
            "retrieval_topk": 10,
            "databases": [
                {
                    "type": "mongo_db",
                    "name": "mongo_db",
                    "host": "localhost",
                    "port": 27017,
                    "database_name": "my_database",
                    "collection_name": "my_collection",
                    "strategies": ["text", "image"]
                },
                {
                    "type": "vector_db",
                    "name": "vector_db",
                    "host": "localhost",
                    "port": 1234,
                    "index_name": "my_index",
                    "strategies": ["vector"]
                }
            ]
        }
        '''
        ...
        
