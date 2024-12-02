import base64
from datetime import datetime
import io
from typing import List, cast
from transformers import AutoModel
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
from numpy.typing import NDArray
from bson import ObjectId

class EmbeddingModel:
    def embed(self, data: RetrievalData) -> List[Vector]:
        raise NotImplementedError("The embed method must be implemented by subclasses.")

class JinnaClip(EmbeddingModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
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
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content])) 
                
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