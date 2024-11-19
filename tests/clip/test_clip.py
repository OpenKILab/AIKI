import unittest
import requests
import base64
from aiki.clip.clip import JinnaClip
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData
from bson import ObjectId
from datetime import datetime

class TestJinnaClip(unittest.TestCase):
    def setUp(self):
        self.clip = JinnaClip()

    def test_text_embedding(self):
        text_data = RetrievalData(
            items=[
                TextModalityData(
                    content="This is a test sentence.",
                    _id=ObjectId(),
                    metadata={"timestamp": int(datetime.now().timestamp()), "summary": "test"}
                )
            ]
        )
        embeddings = self.clip.embed(text_data)
        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)

    def test_image_embedding(self):
        # Load image from local file
        image_path = "resource/source/imgs/外滩人流.png"
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        image_data = RetrievalData(
            items=[
                ImageModalityData(
                    content=base64_image,
                    _id=ObjectId(),
                    metadata={"timestamp": int(datetime.now().timestamp()), "summary": "test"}
                )
            ]
        )
        embeddings = self.clip.embed(image_data)
        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)
        
if __name__ == '__main__':
    unittest.main()