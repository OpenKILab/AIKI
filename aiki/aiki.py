import os
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import MultimodalIndexer
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler, TextModalityData
from aiki.multimodal.vector import VectorHandler
from aiki.retriever.retriever import DenseRetriever
from bson import ObjectId

# TODO: aws s3 SDK
class AIKI:
    def __init__(self, db_name: str = "xiaobu_summary", model_name: str = "lier007/xiaobu-embedding-v2"):
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Optionally, load a local model or take other actions

        embedding_func = model.encode
        processor = MultiModalProcessor()
        source_db = JSONFileDB(f"./db/{db_name}/{db_name}.json")
        chroma_db = ChromaDB(collection_name=f"{db_name}_index", persist_directory=f"./db/{db_name}/{db_name}_index")

        processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
        processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
        processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_func))
        
        self.dense_retriever = DenseRetriever(processor=processor)
        self.multimodal_indexer = MultimodalIndexer(processor=processor)

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
                pick_up_data.append({"url": item.url, "summary": item.metadata.get("summary", "")})
            elif item.modality == ModalityType.TEXT:
                pick_up_data.append(item.content)
        return pick_up_data
        
    def _index(self, data: str):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        if os.path.isfile(data) and data.lower().endswith(image_extensions):
            print(f"Processing image file: {data}")
            retrieval_data = RetrievalData(
                items=[
                    ImageModalityData(
                        url=data,
                        _id=ObjectId(),
                ),
                ]
            )
            self.multimodal_indexer.index(retrieval_data)
        else:
            print(f"Processing non-image data: {data}")
            retrieval_data = RetrievalData(
                items=[
                    TextModalityData(
                        content=data,
                        _id=ObjectId(),
                ),
                ]
            )