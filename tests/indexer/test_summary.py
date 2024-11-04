import pytest
import os
from aiki.config.config import Config
from aiki.indexer.indexer import APISummaryGenerator
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType

@pytest.fixture
def config():
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', '..', 'aiki', 'config', 'config.yaml')
    return Config(config_path)

@pytest.fixture
def api_summary_generator(config):
    return APISummaryGenerator(config)

def test_generate_summary_with_text(api_summary_generator):
    # Create a sample RetrievalData object with text
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.TEXT, content="Example text data")
        ]
    )
    
    # Call the generate_summary method
    summaries = api_summary_generator.generate_summary(retrieval_data)
    
    # Assert that a summary is returned
    assert len(summaries) == 1
    assert isinstance(summaries[0], str)
    assert summaries[0] != ""

'''
def test_generate_summary_with_image(api_summary_generator):
    # Create a sample RetrievalData object with an image
    retrieval_data = RetrievalData(
        items=[
            {"type": RetrievalType.IMAGE, "content": "base64encodedimagestring"}
        ]
    )
    
    # Call the generate_summary method
    summaries = api_summary_generator.generate_summary(retrieval_data)
    
    # Assert that a summary is returned
    assert len(summaries) == 1
    assert isinstance(summaries[0], str)
    assert summaries[0] != ""
    '''