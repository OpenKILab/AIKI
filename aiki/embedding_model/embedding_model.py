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

from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData
from aiki.multimodal.types import Vector
from colpali_engine.models import ColPali, ColPaliProcessor

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
    
class ColPali(EmbeddingModel):
    def __init__(self):
        self.model_name = "vidore/colpali-v1.2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype = torch.bfloat16,
            device_map = self.device,
        ).eval()
        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))
    
    def text_embedding_func(self, data:List) -> List[Vector]:
        batch_queries = self.processor.process_queries(data).to(self.model.device)
        with torch.no_grad():
            data_embeddings = self.model(**batch_queries)
        return data_embeddings
    
    def image_embedding_func(self, data:List) -> List[Vector]:
        # data : base64 encoded image
        images = []
        for item in data:
            base64_string = item
            image_data = base64.b64decode(base64_string)
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            images.append(image)
        batch_images = self.processor.process_queries(images).to(self.model.device)
        image_embeddings = self.model(**batch_images)
        return image_embeddings
    
    def embed(self, data: RetrievalData) -> List[Vector]:
        embeddings = []
        for item in data.items:
            if item.modality == ModalityType.TEXT:
                embeddings.extend(self.text_embedding_func([item.content]))
            elif item.modality == ModalityType.IMAGE:
                embeddings.extend(self.image_embedding_func([item.content])) 
                
        return embeddings
    
    def score(self, query, db_data) -> List[float]: 
        # TODO: define db_data
        query_embeddings = self.text_embedding_func([query])
        db_data_embeddings = self.image_embedding_func(db_data)
        scores = self.processor.score_multi_vector(query_embeddings, db_data_embeddings)
        return scores
    
def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == "__main__":
    # base_path = os.getcwd()

    # print("Current file path:", base_path)
    # file_path = f"{base_path}/resource/source/imgs/外滩小巷.jpg"
    # encoded_image = encode_image_to_base64(file_path)

    # retrieval_data = RetrievalData(
    #     items=[
    #         ImageModalityData(
    #             content= f"""{encoded_image}""",
    #             _id = ObjectId(),
    #             metadata={"timestamp": int((datetime.now()).timestamp()), "summary": "test"}
    #     ),
    #     ]
    # )

    # clip = JinnaClip()

    # print(clip.embed(retrieval_data))
    from typing import cast

    import torch
    from PIL import Image

    from colpali_engine.models import ColPali, ColPaliProcessor

    model_name = "vidore/colpali-v1.2"

    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_name)

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        querry_embeddings = model(**batch_queries)
    print(len(image_embeddings))
    print(len(querry_embeddings))
    scores = processor.score_multi_vector(querry_embeddings, image_embeddings)

