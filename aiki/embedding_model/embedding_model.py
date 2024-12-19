import base64
from datetime import datetime
import io
from typing import List, cast
from transformers import AutoModel, AutoConfig
import os
from dotenv import load_dotenv
import torch
from bson import ObjectId
from PIL import Image
import numpy as np

from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData
from aiki.multimodal.types import Vector
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from bson import ObjectId
from transformers import CLIPProcessor, CLIPModel

class EmbeddingModel:
    def embed(self, data: RetrievalData) -> List[Vector]:
        raise NotImplementedError("The embed method must be implemented by subclasses.")

class JinnaClip(EmbeddingModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(
            'jinaai/jina-clip-v2', 
            trust_remote_code=True, 
        )
        print("============================")
        print(self.device)
        print("============================")
        self.embedding_model.to(self.device)  # Move the model to the specified device
        self.truncate_dim = 512
        self.active_encode()
        
    def active_encode(self):
        self.embedding_model.encode("active yourself", truncate_dim=self.truncate_dim)
    
    def text_embedding_func(self, data:List) -> List:
        return self.embedding_model.encode(data, truncate_dim=self.truncate_dim)
    
    def image_embedding_func(self, data:List) -> List:
        # data : base64 encoded image / bytes image
        imgs = []
        for item in data:
            if isinstance(item, str):
                base64_string = item
                image_data = base64.b64decode(base64_string)
                image_stream = io.BytesIO(image_data)
            elif isinstance(item, bytes):
                image_stream = item
            image = Image.open(image_stream)
            imgs.append(image)
        embeddings = self.embedding_model.encode(imgs, truncate_dim=self.truncate_dim)
        return embeddings

    def batch_image_embedding_func(self, data:List) -> List:
        return self.embedding_model.encode(data, truncate_dim=self.truncate_dim)
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content])) 
                
        return embeddings

    def batch_embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = [None] * len(data.items)  # Initialize with placeholders
        image_data = [(i, item.url) for i, item in enumerate(data.items) if item.modality == ModalityType.IMAGE]
        text_data = [(i, item.content) for i, item in enumerate(data.items) if item.modality == ModalityType.TEXT]
        
        # Process image embeddings
        if image_data:
            indices, contents = zip(*image_data)
            image_embeddings = self.batch_image_embedding_func(list(contents))
            for idx, embedding in zip(indices, image_embeddings):
                embeddings[idx] = embedding

        # Process text embeddings
        if text_data:
            indices, contents = zip(*text_data)
            text_embeddings = self.text_embedding_func(list(contents))
            for idx, embedding in zip(indices, text_embeddings):
                embeddings[idx] = embedding

        return embeddings
    
class ColPaliModel(EmbeddingModel):
    def __init__(self):
        self.model_name = "vidore/colpali-v1.2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit quantization
            quantization_method="bitsandbytes",  # Use bitsandbytes library
            llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 CPU offload
        )
        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype = torch.bfloat16,
            quantization_config=self.quantization_config,
            device_map=self.device,
        ).eval()
        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))
    
    def text_embedding_func(self, data:List) -> torch.Tensor:
        batch_queries = self.processor.process_queries(data).to(self.device)
        with torch.no_grad():
            data_embeddings = self.model(**batch_queries)
        return data_embeddings
    
    def image_embedding_func(self, data:List) -> torch.Tensor:
        # data : base64 encoded image
        images = []
        for item in data:
            base64_string = item
            image_data = base64.b64decode(base64_string)
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            images.append(image)
        batch_images = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        return image_embeddings
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content]))
        embeddings = [embedding.to(torch.float32).cpu().numpy() for embedding in embeddings]
        return embeddings
    
    def score(self, query: RetrievalData, embedding_source_data: List[Vector]) -> List[List[float]]: 
        query_embeddings = torch.tensor(self.embed(query), dtype=torch.float32, device=self.device)
        
        torch_embedding_source_data = [torch.tensor(vector, dtype=torch.float32, device=self.device) for vector in embedding_source_data]
        scores = self.processor.score_multi_vector(query_embeddings, torch_embedding_source_data)
        return scores.tolist()
    
class VitClip(EmbeddingModel):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def text_embedding_func(self, data:List) -> List:
        inputs = self.processor(text=data, return_tensors="pt", padding=True)
        # tensor
        outputs = self.model.get_text_features(**inputs)
        outputs = outputs.tolist()
        return outputs
    
    def image_embedding_func(self, data:List) -> List:
        # data : base64 encoded image / bytes image
        imgs = []
        for item in data:
            if isinstance(item, str):
                base64_string = item
                image_data = base64.b64decode(base64_string)
                image_stream = io.BytesIO(image_data)
            elif isinstance(item, bytes):
                image_stream = item
            image = Image.open(image_stream)
            imgs.append(image)
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True)
        # tensor
        outputs = self.model.get_image_features(**inputs)
        outputs = outputs.tolist()
        return outputs
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content])) 
                
        return embeddings
    
    def batch_embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = [None] * len(data.items)  # Initialize with placeholders
        image_data = [(i, item.url) for i, item in enumerate(data.items) if item.modality == ModalityType.IMAGE]
        text_data = [(i, item.content) for i, item in enumerate(data.items) if item.modality == ModalityType.TEXT]
        
        # Process image embeddings
        if image_data:
            indices, contents = zip(*image_data)
            image_embeddings = self.batch_image_embedding_func(list(contents))
            for idx, embedding in zip(indices, image_embeddings):
                embeddings[idx] = embedding

        # Process text embeddings
        if text_data:
            indices, contents = zip(*text_data)
            text_embeddings = self.text_embedding_func(list(contents))
            for idx, embedding in zip(indices, text_embeddings):
                embeddings[idx] = embedding

        return embeddings
        
    
def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == "__main__":
    colpali = ColPaliModel()
    
    import base64
    from io import BytesIO

    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
    ]
    
    def encode_image_to_base64(image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="jpeg")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    base64_images = [encode_image_to_base64(image) for image in images]
    rd = RetrievalData(
        items= [
            ImageModalityData(
                url= file_path,
                _id = ObjectId(),
                content = base64_images[0],
            ),
            TextModalityData(
                _id = ObjectId(),
                content = "test",
            )
        ]
    )

    embeddings = colpali.embed(rd)
    query_rd = RetrievalData(
        items = [
            TextModalityData(
                _id = ObjectId(),
                content = "content",
            ),
            TextModalityData(
                _id = ObjectId(),
                content = "test",
            )
        ]
    )
    print(colpali.score(query_rd, [embeddings[0], embeddings[1]]))