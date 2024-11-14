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
        if item.__class__ not in [TextModalityData, ImageModalityData]:
            raise ValueError(f"{self.__class__.__name__}.genearte_summary(). There is no such modal data processing method")
        
        content_type = "image_url" if item.__class__ == ImageModalityData else "text"
        content_value = {
            "url": f"data:image/jpeg;base64,{item._content}"
        } if item.__class__ == ImageModalityData else item.content
        
        prompt_text = "What is in this image?" if item.__class__ == ImageModalityData else "Please summarize this text."
        
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
    def __init__(self, model_path, sourcedb: BaseKVDatabase, vectordb: BaseVectorDatabase, processor: MultiModalProcessor = MultiModalProcessor(), chunker: BaseChunker = FixedSizeChunker()):
        self.model_path = model_path
        self.sourcedb = sourcedb  # Source database storage
        self.vectordb = vectordb  # Vector database storage
        self.chunker = chunker
        
        self.processor = processor
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(BaseIndexer):
    def __init__(self, model_path, sourcedb: BaseKVDatabase = None, vectordb: BaseVectorDatabase = None, chunker: BaseChunker = FixedSizeChunker(), processor = None):
        super().__init__(model_path, sourcedb, vectordb, chunker=chunker, processor=processor)
        try:
            self.processor._get_handler(ModalityType.TEXT)
        except ValueError:
            self.processor.register_handler(ModalityType.TEXT, TextHandler(database=self.sourcedb))
        try:
            self.processor._get_handler(ModalityType.VECTOR)
        except ValueError:
            self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=self.vectordb, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
        
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
                self.processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id=cur_id, content=data, metadata={"summary": "","timestamp": data.metadata["timestamp"]})])
                self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=cur_id, content=data)])
                
class ImageIndexer(BaseIndexer):
    def __init__(self, model_path, processor = None, sourcedb: BaseKVDatabase = None, vectordb: BaseVectorDatabase = None, chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator()):
        super().__init__(model_path, sourcedb, vectordb, chunker=chunker,  processor = processor)
        self.summary_generator = summary_generator
        try:
            self.processor._get_handler(ModalityType.IMAGE)
        except ValueError:
            self.processor.register_handler(ModalityType.IMAGE, TextHandler(database=self.sourcedb))
        try:
            self.processor._get_handler(ModalityType.VECTOR)
        except ValueError:
            self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=self.vectordb, embedding_func=embedding_functions.DefaultEmbeddingFunction()))
        
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
            self.processor.execute_operation(ModalityType.IMAGE, ImageHandlerOP.MSET, [ImageModalityData(_id=id, _content=retrieval_data._content, metadata={"summary": summary, "timestamp": retrieval_data.metadata["timestamp"], "parent": [], "children": []})])
            image_data = ImageModalityData(_id=id, _content=summary)
            self.processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [image_data])

class MultimodalIndexer(BaseIndexer):
    def __init__(self, model_path, sourcedb: BaseKVDatabase = None, vectordb: BaseVectorDatabase = None, processor: MultiModalProcessor = MultiModalProcessor(), chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator()):
        super().__init__(model_path, sourcedb, vectordb)
        self.text_indexer = TextIndexer(model_path,  processor = processor, chunker=chunker)
        self.image_indexer = ImageIndexer(model_path,  processor = processor, chunker=chunker, summary_generator=summary_generator)
    
    def index(self, data: RetrievalData):
        text_retrieval_data = RetrievalData(items=[])
        image_retrieval_data = RetrievalData(items=[])
        for retrieval_data in data.items:
            if retrieval_data.__class__ == TextModalityData:
                text_retrieval_data.items.append(
                    retrieval_data
                )
            elif retrieval_data.__class__ == ImageModalityData:
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

import torch
from transformers import AutoModel

# Example usage
if __name__ == "__main__":
    processor = MultiModalProcessor()
    source_db = JSONFileDB("./db/flicker8k_jina.json")
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/flicker8k_jina_index")
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db,
                                                                  embedding_func=model.encode))

    multimodal_indexer = MultimodalIndexer(model_path='path/to/model', processor=processor)
    
    base_path = os.getcwd()

    print("Current file path:", base_path)
    file_path = f"{base_path}/resource/source/imgs/外滩小巷.jpg"
    encoded_image = encode_image_to_base64(file_path)
    
    retrieval_data = RetrievalData(
        items=[
            ImageModalityData(
                _content= encoded_image,
                _id = ObjectId(),
                metadata={"timestamp": datetime.now() - timedelta(days = 7), "summary": "test"}
        ),
        ]
    )

    # Index the data
    multimodal_indexer.index(retrieval_data)
