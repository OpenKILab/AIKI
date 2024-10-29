from typing import List, Dict, Any
from aiki.modal.retrieval_data import RetrievalData

class BaseReranker:
    def __init__(self, config):
        self.config = config
        
    def rerank(self, query: RetrievalData, data_list: RetrievalData, topk: int = None) -> RetrievalData:
        ...
