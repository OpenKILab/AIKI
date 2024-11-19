import pytest
import os
import base64

from aiki.indexer.indexer import APISummaryGenerator
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType


@pytest.fixture
def api_summary_generator():
    return APISummaryGenerator()

def test_generate_summary_with_text(api_summary_generator):
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.TEXT, content="Example text data")
        ]
    )
    
    summaries = api_summary_generator.generate_summary(retrieval_data)
    
    assert len(summaries) == 1
    assert isinstance(summaries[0], str)
    assert summaries[0] != ""

def test_generate_summary_with_image(api_summary_generator):
    # Create a sample RetrievalData object with an image
    
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, '..', '..', 'aiki', 'resource', 'source', 'imgs', '外滩路人.jpg')

    base64_image = encode_image(image_path)
    
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.IMAGE, content=base64_image)
        ]
    )
    
    summaries = api_summary_generator.generate_summary(retrieval_data)
    assert len(summaries) == 1
    assert isinstance(summaries[0], str)
    assert summaries[0] != ""