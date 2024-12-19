import asyncio
from datetime import datetime
import os
from typing import List, Union
from aiki.database.sqlite import SQLiteDB
from aiki.embedding_model.embedding_model import JinnaClip, VitClip
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from aiki.database.milvus import MilvusDB
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import ClipIndexer, MultimodalIndexer
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler, TextModalityData
from aiki.multimodal.vector import VectorHandler
from aiki.retriever.retriever import DenseRetriever
from bson import ObjectId
from transformers import CLIPModel
import logging

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)

# TODO: aws s3 SDK
class AIKI:
    def __init__(self, db_path: str = "./db/", model_name: str = "openai/clip-vit-base-patch32"):
        self.load_config()
        ## TODO:  summary 和 clip 间的切换
        try:
            self.model = JinnaClip()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Optionally, load a local model or take other actions

        embedding_func = self.model.embed
        self.processor = MultiModalProcessor()
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        source_db = SQLiteDB(os.path.join(db_path, "source"))
        chroma_db = ChromaDB(collection_name=f"index", persist_directory=os.path.join(db_path, "index"))

        self.processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
        self.processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
        self.processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_func))
        
        self.dense_retriever = DenseRetriever(processor=self.processor, embedding_model = JinnaClip())
        self.multimodal_indexer = ClipIndexer(processor=self.processor)
        
    async def _start_processor_worker(self):
        await self.processor.start_worker()
    
    def load_config(self):
        # ignore log
        libraries = ["urllib3", "aiosqlite", "tortoise", "asyncio", "httpx", "chromadb", "botocore", "boto3"]
        for lib in libraries:
            logging.getLogger(lib).setLevel(logging.WARNING)

    def index(self, data: Union[str, List[str]]):
        with tqdm(total=1, desc="Indexing Process") as pbar:
            if isinstance(data, str):
                if os.path.isdir(data):
                    data = [os.path.join(data, file) for file in os.listdir(data)]
                    for item in data:
                        self._index(item)
                    
                else:
                    self._index(data)
                
                pbar.update(1)
                return
            
            if isinstance(data, list):
                for item in tqdm(data, desc="Indexing Progress"):
                    self._index(item)
                pbar.update(1)
                return
            else:
                raise ValueError("Data must be a string or a list of strings")
            pbar.update(1)

    def retrieve(self, data: str, num:int = 4):
        query_data = RetrievalData(
            items=[
                    TextModalityData(
                        content=data,
                        _id=ObjectId(),
                ),
            ]
        )
        result_data = self.dense_retriever.search(query_data, num=num)
        pick_up_data = []
        for item in result_data.items:
            if item.modality == ModalityType.IMAGE:
                pick_up_data.append({"id": item._id, "url": item.url, "summary": item.metadata.get("summary", "")})
            elif item.modality == ModalityType.TEXT:
                pick_up_data.append({"id": item._id, "content": item.content})
        return pick_up_data
        
    def _index(self, data: str):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        retrieval_data = None
        if os.path.isfile(data) and data.lower().endswith(image_extensions):
            print(f"Processing image file: {data}")
            retrieval_data = RetrievalData(
                items=[
                    ImageModalityData(
                        url=data,
                        _id=ObjectId(),
                        metadata={
                            "timestamp": int(datetime.now().timestamp()),
                        }
                ),
                ]
            )
        else:
            print(f"Processing non-image data: {data}")
            retrieval_data = RetrievalData(
                items=[
                    TextModalityData(
                        content=data,
                        _id=ObjectId(),
                        metadata={
                            "timestamp": int(datetime.now().timestamp()),
                        }
                ),
                ]
            )
        self.multimodal_indexer.index(retrieval_data)
        
    def batch_index(self, data: RetrievalData):
        self.multimodal_indexer.batch_index(data)