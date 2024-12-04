from datetime import datetime
import json
from platform import processor

from bson import ObjectId
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import MultimodalIndexer
from aiki.modal import retrieval_data
from aiki.modal.retrieval_data import RetrievalData, RetrievalType
from aiki.multimodal.base import BaseModalityData, ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler, TextModalityData
from aiki.multimodal.vector import VectorHandler

from sentence_transformers import SentenceTransformer

from aiki.retriever.retriever import DenseRetriever

class RAGAgentBridge:
    def __init__(self, name: str, embedding_func: callable=None):
        self.name = name
        if embedding_func is None:
            self.model = SentenceTransformer('lier007/xiaobu-embedding-v2', device="cpu")
            self.embedding_func = self.model.encode
        else:
            self.embedding_func = embedding_func
        
        self.processor = MultiModalProcessor()
        self.source_db = JSONFileDB(f"./db/{name}/{name}.json")
        self.vector_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"./db/{name}/{name}_index")

        self.processor.register_handler(ModalityType.TEXT, TextHandler(database=self.source_db))
        self.processor.register_handler(ModalityType.IMAGE, TextHandler(database=self.source_db))
        self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=self.vector_db, embedding_func=self.embedding_func))
        
        self.multimodal_indexer = MultimodalIndexer(processor=self.processor)
        self.multimodal_retriever = DenseRetriever(processor=self.processor)
        
    def _get_modality_data_class(self, query: str):
        modality_map = {
            "data:image": ImageModalityData,
        }
        for prefix, modality_class in modality_map.items():
            if query.startswith(prefix):
                return modality_class
        return TextModalityData
        
    def query(self, retrieval_data: RetrievalData) -> RetrievalData:
        items = []
        for item in retrieval_data.items:
            query = item.content
            modal_class: BaseModalityData = self._get_modality_data_class(query)
            res_item = modal_class(
                        _id = ObjectId(),
                        content = query,
                        metadata=item.metadata
                    )
            items.append(res_item)
        return self.multimodal_retriever.search(
            query = RetrievalData(
            items=items,
        ),
            num = 4,)
    
    def add(self, retrieval_data: RetrievalData) -> RetrievalData:
        items = []
        for item in retrieval_data.items:
            query = item.content
            modal_class: BaseModalityData = self._get_modality_data_class(query)
            res_item = modal_class(
                        _id = ObjectId(),
                        content = query,
                        metadata=item.metadata
                    )
            items.append(res_item)
        index_retrieval_data = RetrievalData(
            items=items
        )
        self.multimodal_indexer.index(index_retrieval_data)
        return index_retrieval_data
        
        
    def delete():
        ...
        
    def update():
        ...
    