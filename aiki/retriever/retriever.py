from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Protocol

from bson import ObjectId
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.modal.retrieval_data import KVSchema, RetrievalData, RetrievalType, RetrievalItem
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageHandlerOP
from aiki.multimodal.text import TextHandler, TextHandlerOP, TextModalityData
from aiki.multimodal.vector import VectorHandler, VectorHandlerOP
from chromadb.utils import embedding_functions

class DataPool:
    def __init__(self):
        self.pool = {}

    def add(self, component_name: str, data: RetrievalData):
        if component_name not in self.pool:
            self.pool[component_name] = []
            
        self.pool[component_name].append(data)
    
    def get(self, component_name: str) -> List[RetrievalData]:
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
    def __init__(self, processor: MultiModalProcessor, preretriever_components: List[PreRetrieverComponent] = None, postretriever_components: List[PostRetrieverComponent] = None, recall_strategies: List[RecallStrategy] = None):
        self.topk = 4
        self.recall_strategies = recall_strategies or []
        self.database = {}
        self.data_pool = DataPool()
        self.preretriever_components = []
        self.postretriever_components = []
        self.processor = processor
        
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

    def search(self, query: RetrievalData, num: int = None) -> RetrievalData:
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
    def __init__(self, processor):
        super().__init__(processor)
        
    def _search(self, query: RetrievalData, num: int = 10):
        queries = [q.content for q in query.items if q.__class__ == TextModalityData]
        start_time = query.items[0].metadata.get("start_time", 0)
        end_time = query.items[0].metadata.get("end_time", int((datetime.now()).timestamp()))
        vector_db_result = self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, queries, top_k=num, start_time = start_time, end_time = end_time)
        for item in vector_db_result:
            for (_id, modality_type) in item:
                operation = TextHandlerOP.MGET
                if modality_type == ModalityType.IMAGE:
                    operation = ImageHandlerOP.MGET
                elif modality_type == ModalityType.TEXT:
                    operation = TextHandlerOP.MGET
                self.data_pool.add("_search", self.processor.execute_operation(modality_type, operation, [str(_id)])[0])

    def search(self, query: RetrievalData, num: int = 10) -> RetrievalData:
        self._search(query, num)
        search_res = self.data_pool.get("_search")
        return RetrievalData(
            items = [
                item for item in search_res
                ]
            )
        
if __name__ == "__main__":
    processor = MultiModalProcessor()
    source_db = JSONFileDB("./db/flicker8k.json")
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/flicker8k_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
    
    dense_retriever = DenseRetriever(processor=processor)
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            content= "棕色狗狗",
            _id = ObjectId(),
            metadata={
                "start_time": 0,
                "end_time": int(datetime.now().timestamp())
            }
        )
    ])
    result = dense_retriever.search(retrieval_data, num=10)
    for r in result.items:
        print(r.metadata["summary"])