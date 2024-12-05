import base64
from datetime import datetime
import io
from typing import List
from transformers import AutoModel, AutoConfig
import os
from dotenv import load_dotenv
import torch
from bson import ObjectId
from PIL import Image

from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData
from aiki.multimodal.types import Vector

class EmbeddingModel:
    def embed(self, data: RetrievalData) -> List[Vector]:
        raise NotImplementedError("The embed method must be implemented by subclasses.")

class JinnaClip(EmbeddingModel):
    def __init__(self):
        self.device = torch.device("cuda")
        self.embedding_model = AutoModel.from_pretrained(
            'jinaai/jina-clip-v2', 
            trust_remote_code=True, 
            device_map=self.device
        )
        self.truncate_dim = 512
    
    def text_embedding_func(self, data:List) -> List:
        embeddings = []
        for item in data:
            embeddings.append(self.embedding_model.encode_text(item, truncate_dim=self.truncate_dim))
        return embeddings
    
    def image_embedding_func(self, data:List) -> List:
        # data : base64 encoded image
        embeddings = []
        for item in data:
            base64_string = item
            image_data = base64.b64decode(base64_string)
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            embeddings.append(self.embedding_model.encode_image(image, truncate_dim=self.truncate_dim))
            
        return embeddings

    def batch_image_embedding_func(self, data:List) -> List:
        return self.embedding_model.encode_image(data, truncate_dim=self.truncate_dim)
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content])) 
                
        return embeddings

    def batch_embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        urls = []
        for item in data.items:
            urls.append(item.url)
        embeddings = self.batch_image_embedding_func(urls)
        return embeddings
    
def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == "__main__":
    base_path = os.getcwd()

    print("Current file path:", base_path)
    file_path = f"{base_path}/resource/source/imgs/外滩小巷.jpg"
    encoded_image = encode_image_to_base64(file_path)

    retrieval_data = RetrievalData(
        items=[
            ImageModalityData(
                url= file_path,
                _id = ObjectId(),
                metadata={"timestamp": int((datetime.now()).timestamp()), "summary": "test"}
        ),
        ]
    )

    clip = JinnaClip()

    print(clip.embed(retrieval_data))
