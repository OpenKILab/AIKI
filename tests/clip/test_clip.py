# test_clip.py
import io
from PIL import Image
import pytest
from aiki.clip.clip import JinnaClip
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData
from bson import ObjectId
import base64
from datetime import datetime

def test_text_embedding():
    clip = JinnaClip()
    text_data = RetrievalData(
        items=[
            TextModalityData(
                content="This is a test sentence.",
                _id=ObjectId(),
                metadata={"timestamp": int((datetime.now()).timestamp()), "summary": "test"}
            ),
        ]
    )
    embeddings = clip.embed(text_data)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == clip.truncate_dim

def test_image_embedding():
    clip = JinnaClip()
    # Create a simple black square image and encode it to base64
    image = Image.new('RGB', (10, 10), color='black')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    image_data = RetrievalData(
        items=[
            ImageModalityData(
                content=encoded_image,
                _id=ObjectId(),
                metadata={"timestamp": int((datetime.now()).timestamp()), "summary": "test"}
            ),
        ]
    )
    embeddings = clip.embed(image_data)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == clip.truncate_dim

if __name__ == "__main__":
    pytest.main()