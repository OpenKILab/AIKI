import base64
from typing import List
from openai import OpenAI
from aiki.config.config import Config
from bson import ObjectId
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from aiki.database import BaseKVDatabase, BaseVectorDatabase
from aiki.database import JSONFileDB
from aiki.database.chroma import ChromaDB
from aiki.indexer.chunker import BaseChunker, FixedSizeChunker
from aiki.modal.retrieval_data import KVSchema, RetrievalData, RetrievalItem, RetrievalType

import os

from aiki.multimodal import ModalityType, MultiModalProcessor, TextHandler, TextHandlerOP, TextModalityData, VectorHandler, VectorHandlerOP
from chromadb.utils import embedding_functions

from aiki.multimodal.image import ImageHandlerOP, ImageModalityData

# 多模态数据生成文本摘要
class BaseSummaryGenerator(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        
    def generate_summary(self, data: RetrievalItem):
        ...
        
class ModelSummaryGenerator(BaseSummaryGenerator):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = self.load_model(self.model_path) 
    
    def load_model(self, model_path):
        # load tokenizer and model
        ...
        
    def generate_summary(self, data: RetrievalItem):
        ...

class APISummaryGenerator(BaseSummaryGenerator):
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, '..', '..', 'aiki', 'config', 'config.yaml')
        config = Config(config_path)
        
        super().__init__(config.get('model_path', 'default_model_path'))
        self.client = OpenAI(
            base_url=config.get('base_url', "https://api.claudeshop.top/v1")
        )
        self.model = config.get('model', "gpt-4o-mini")
        
    def generate_summary(self, data: RetrievalItem) -> str:
        item = data
        if item.modality not in [ModalityType.TEXT, ModalityType.IMAGE]:
            raise ValueError(f"{self.item.modality}.genearte_summary(). There is no such modal data processing method")
        
        content_type = "image_url" if item.modality == ModalityType.IMAGE else "text"
        content_value = {
            "url": f"data:image/jpeg;base64,{item.content}"
        } if item.modality == ModalityType.TEXT else item.content
        
        prompt_text = "What is in this image?" if item.modality == ModalityType.IMAGE else "Please summarize this text."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": content_type,
                            content_type: content_value,
                        },
                    ],
                }
            ],
        )
        summary = response.choices[0].message.content
        return summary

class BaseIndexer(ABC):
    def __init__(self, processor: MultiModalProcessor = MultiModalProcessor(), chunker: BaseChunker = FixedSizeChunker(), model_path: str = None):
        self.model_path = model_path
        self.chunker = chunker
        
        self.processor = processor
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(BaseIndexer):
    def __init__(self, chunker: BaseChunker = FixedSizeChunker(), processor: MultiModalProcessor = None, model_path: str = None):
        super().__init__(chunker=chunker, processor=processor, model_path=model_path)
        
    def index(self, data: RetrievalData):
        for retreval_data in data.items:
            if retreval_data.__class__ != TextModalityData:
                raise ValueError(f"{self.__class__.__name__}.index(). Unsupported data type: {retreval_data.__class__.__name__}")
            '''
            # parent data
            id = ObjectId()
            dataSchema = KVSchema(
                _id=id,
                modality="text",
                summary="",
                source_encoded_data=retreval_data.content,
                inserted_timestamp=datetime.now(),
                parent=[],
                children=[]
            )
            self.processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id=id, content=data)])
            '''
            chunks = self.chunker.chunk(retreval_data.content)
            for data in chunks:
                cur_id = ObjectId()
                # dataSchema = KVSchema(
                #     _id=cur_id,
                #     modality="text",
                #     summary="",
                #     source_encoded_data=data,
                #     inserted_timestamp=datetime.now(),
                #     parent=[id],
                #     children=[]
                # )
                self.processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id=cur_id, content=data, metadata={"summary": "","timestamp": retreval_data.metadata["timestamp"]})])
                self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=cur_id, content=data)])
                
class ImageIndexer(BaseIndexer):
    def __init__(self, processor: MultiModalProcessor = None, chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator(), model_path: str = None):
        super().__init__(chunker=chunker, processor = processor, model_path=model_path)
        self.summary_generator = summary_generator

    def index(self, data: RetrievalData):
        for retrieval_data in data.items:
            if retrieval_data.__class__ != ImageModalityData:
                raise ValueError(f"{self.__class__.__name__}.index(). Unsupported data type: {retrieval_data.__class__.__name__}")
            id = ObjectId()
            if "summary" not in retrieval_data.metadata:
                summary = self.summary_generator.generate_summary(retrieval_data)
            else:
                summary = retrieval_data.metadata["summary"]
            # dataSchema = KVSchema(
            #     _id=id,
            #     modality="image",
            #     summary=summary,
            #     source_encoded_data=retreval_data.content,
            #     inserted_timestamp=datetime.now(),
            #     parent=[],
            #     children=[]
            # )
            self.processor.execute_operation(ModalityType.IMAGE, ImageHandlerOP.MSET, [ImageModalityData(_id=id, content=retrieval_data.content, metadata={"summary": summary, "timestamp": retrieval_data.metadata["timestamp"], "parent": [], "children": []})])
            image_data = ImageModalityData(_id=id, content=summary, metadata={"timestamp": retrieval_data.metadata["timestamp"]})
            self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [image_data])

class MultimodalIndexer(BaseIndexer):
    def __init__(self, processor: MultiModalProcessor = MultiModalProcessor(), chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator(), model_path: str = None):
        super().__init__(processor=processor, model_path=model_path)
        self.text_indexer = TextIndexer(chunker=chunker, processor = processor, model_path=model_path)
        self.image_indexer = ImageIndexer(chunker=chunker, summary_generator=summary_generator, processor = processor, model_path=model_path)
    
    def index(self, data: RetrievalData):
        text_retrieval_data = RetrievalData(items=[])
        image_retrieval_data = RetrievalData(items=[])
        for retrieval_data in data.items:
            if retrieval_data.modality == ModalityType.TEXT:
                text_retrieval_data.items.append(
                    retrieval_data
                )
            elif retrieval_data.modality == ModalityType.IMAGE:
                image_retrieval_data.items.append(
                    retrieval_data
                )
            else:
                raise ValueError(f"Unsupported data type: {retrieval_data.__class__.__name__}")
        self.text_indexer.index(text_retrieval_data)
        self.image_indexer.index(image_retrieval_data)
        
class KnowledgeGraphIndexer(BaseIndexer):
    ...
    
def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Example usage
if __name__ == "__main__":
    processor = MultiModalProcessor()
    source_db = JSONFileDB("./db/test/test.json")
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/test/test_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
    
    multimodal_indexer = MultimodalIndexer(processor=processor)
    
    base_path = os.getcwd()

    print("Current file path:", base_path)
    file_path = f"{base_path}/resource/source/imgs/外滩小巷.jpg"
    encoded_image = encode_image_to_base64(file_path)
    
    retrieval_data = RetrievalData(
        items=[
            ImageModalityData(
                content= f""" content """,
                _id = ObjectId(),
                metadata={"timestamp": int((datetime.now() - timedelta(days = 7)).timestamp()), "summary": "test"}
        ),
        ]
    )

    # Index the data
    multimodal_indexer.index(retrieval_data)
