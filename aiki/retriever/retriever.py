from abc import ABC
from typing import List, Dict, Protocol
from aiki.corpus.mockdatabase import DatabaseConnectionFactory
from aiki.modal.retrieval_data import RetrievalData, RetrievalType, RetrievalItem

class RecallStrategy(Protocol):
    def search(self, query: RetrievalData, num: int) -> RetrievalData:
        """Search for relevant data based on the strategy."""
        ...

class BaseRetriever(ABC):
    def __init__(self, config, recall_strategies: List[RecallStrategy] = None):
        self.topk = config.get("retrieval_topk", 4)
        self.recall_strategies = recall_strategies or []
        self.config = config
        self.database = {}
        self.load_config()
    
    def load_config(self):
        for db_config in self.config.get("databases", []):
            db_type = db_config.get("type")
            db_args = db_config.get("args")
            if db_type:
                connection = DatabaseConnectionFactory.create_connection(db_type=db_type, **db_args)
                self.database[db_type] = connection

    def pre_retrieve(self, query: RetrievalData) -> RetrievalData:
        """Pre-process the query before searching."""
        ...

    def post_retrieve(self, query: RetrievalData, results: RetrievalData) -> RetrievalData:
        """Post-process the results after searching."""
        ...

    def _search(self, query: RetrievalData, num: int = None) -> RetrievalData:
        ...

    def search(self, query: RetrievalData, num: int = None) -> RetrievalData:
        """Retrieve topk relevant data in corpus."""
        query = self.pre_retrieve(query)
        results = self._search(query, num)
        results = self.post_retrieve(query, results)
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
    dense_retriever.search(retrieval_data)