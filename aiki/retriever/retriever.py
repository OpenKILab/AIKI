from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Protocol

from bson import ObjectId
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.embedding_model.embedding_model import EmbeddingModel, JinnaClip
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
        return self.pool.get(component_name, [])
    
    def clear(self, component_name: str):
        self.pool[component_name] = []

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
    def __init__(
        self, 
        processor: MultiModalProcessor, 
        preretriever_components: List[PreRetrieverComponent] = None, 
        postretriever_components: List[PostRetrieverComponent] = None, 
        recall_strategies: List[RecallStrategy] = None, 
        embedding_model: EmbeddingModel = None
    ):
        self.topk = 4
        self.recall_strategies = recall_strategies or []
        self.database = {}
        self.data_pool = DataPool()
        self.preretriever_components = preretriever_components or []
        self.postretriever_components = postretriever_components or []
        self.processor = processor
        self.embedding_model = embedding_model
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
    def __init__(
        self, 
        processor: MultiModalProcessor, 
        embedding_model: EmbeddingModel=None
        ):
        super().__init__(processor, embedding_model)
        self.processor = processor
        self.embedding_model = embedding_model
        
    def _search(self, query: RetrievalData, num: int = 10):
        queries = [q.content for q in query.items if q.__class__ == TextModalityData]
        start_time = 0
        end_time = int((datetime.now()).timestamp())
        if query.items[0].metadata:
            start_time = query.items[0].metadata.get("start_time", 0)
            end_time = query.items[0].metadata.get("end_time", int((datetime.now()).timestamp()))
        if self.embedding_model:
            query_embeddings = self.embedding_model.embed(query)
            vector_db_result = self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, query_embeddings=query_embeddings, top_k=num, start_time = start_time, end_time = end_time)
        else:
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
        self.data_pool.clear("_search")
        result = RetrievalData(
            items = [
                item for item in search_res
                ]
            )
        return result
        
if __name__ == "__main__":
    processor = MultiModalProcessor()
    name = "jina_clip"
    source_db = JSONFileDB(f"./db/{name}/{name}.json")
    chroma_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"./db/{name}/{name}_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
    
    dense_retriever = DenseRetriever(processor=processor, embedding_model = JinnaClip())
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            content= "棕色狗狗",
            _id = ObjectId(),
            metadata={
                "start_time": 1324235,
                "end_time": int(datetime.now().timestamp())
            }
        )
    ])
    result = dense_retriever.search(retrieval_data, num=10)
    for item in result.items:
        print(item.metadata["summary"])
        