from datetime import datetime
import json
from platform import processor

from bson import ObjectId
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import MultimodalIndexer
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
            self.model = SentenceTransformer('lier007/xiaobu-embedding-v2')
            self.embedding_func = self.model.encode
        else:
            self.embedding_func = embedding_func
        
        self.processor = MultiModalProcessor()
        self.source_db = JSONFileDB(f"./db/{name}/{name}.json")
        self.vector_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"./db/{name}/test_index")

        self.processor.register_handler(ModalityType.TEXT, TextHandler(database=self.source_db))
        self.processor.register_handler(ModalityType.IMAGE, TextHandler(database=self.source_db))
        self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=self.vector_db, embedding_func=self.embedding_func))
        
        self.multimodal_indexer = MultimodalIndexer(processor=processor)
        self.multimodal_retriever = DenseRetriever(processor=processor)
        
    def _get_modality_data_class(self, query: str):
        modality_map = {
            "data:image": ImageModalityData,
        }
        for prefix, modality_class in modality_map.items():
            if query.startswith(prefix):
                return modality_class
        return TextModalityData
        
    def query(self, retrieval_data: str) -> RetrievalData:
        # TODO: decode retrieval_data
        query = "xxx"
        modal_class: BaseModalityData = self._get_modality_data_class(query)
        retrieval_data = RetrievalData(
            items=[
                modal_class(
                    content = query,
                    _id = ObjectId(),
                    metadata={int((datetime.now()).timestamp())}
                )
            ]
        )
        return self.multimodal_retriever.search(retrieval_data)
    
    def add(self, retrieval_data: str) -> RetrievalData:
        # TODO: decode retrieval_data
        query = "xxx"
        modal_class: BaseModalityData = self._get_modality_data_class(query)
        retrieval_data = RetrievalData(
            items=[
                modal_class(
                    content = query,
                    _id = ObjectId(),
                    metadata={int((datetime.now()).timestamp())}
                )
            ]
        )
        return self.multimodal_indexer.index(retrieval_data)
        
    def delete():
        ...
        
    def update():
        ...
    