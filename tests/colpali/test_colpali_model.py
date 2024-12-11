# test_colpali_model.py

import pytest
import torch
from aiki.embedding_model.embedding_model import ColPaliModel
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.text import TextModalityData
from aiki.multimodal.image import ImageModalityData
from bson import ObjectId
import base64
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="jpeg")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@pytest.fixture
def colpali_model():
    model = ColPaliModel()
    # No need to call model.to(device) since it's handled internally
    return model


@pytest.fixture
def sample_retrieval_data():
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    base64_images = [encode_image_to_base64(image) for image in images]
    return RetrievalData(
        items=[
            ImageModalityData(
                _id=ObjectId(),
                content=base64_images[0],
            ),
            TextModalityData(
                _id=ObjectId(),
                content="test",
            )
        ]
    )

def test_embed(colpali_model, sample_retrieval_data):
    embeddings = colpali_model.embed(sample_retrieval_data)
    assert len(embeddings) == 2
    # assert all(isinstance(embedding, torch.Tensor) for embedding in embeddings)


def test_embed(colpali_model, sample_retrieval_data):
    embeddings = colpali_model.embed(sample_retrieval_data)
    assert len(embeddings) == 2

def test_score(colpali_model, sample_retrieval_data):
    embeddings = colpali_model.embed(sample_retrieval_data)
    query_rd = RetrievalData(
        items=[
            TextModalityData(
                _id=ObjectId(),
                content="content",
            ),
            TextModalityData(
                _id=ObjectId(),
                content="test",
            )
        ]
    )
    scores = colpali_model.score(query_rd, [embeddings[0], embeddings[1]])
    assert isinstance(scores, list)
    assert len(scores) == 2
    assert all(isinstance(score, list) for score in scores)