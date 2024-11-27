import base64
import os
import shutil
import pytest
from unittest.mock import MagicMock

import requests
from aiki.indexer.indexer import TextIndexer, ImageIndexer, MultimodalIndexer
from aiki.modal.retrieval_data import RetrievalData, TextModalityData, ImageModalityData
from aiki.multimodal import ModalityType, MultiModalProcessor
from aiki.indexer.chunker import FixedSizeChunker

@pytest.fixture
def mock_processor():
    processor = MagicMock(spec=MultiModalProcessor)
    return processor

@pytest.fixture
def text_data():
    return RetrievalData(items=[
        TextModalityData(content="Sample text", _id="text_id", metadata={"timestamp": 1234567890})
    ])

@pytest.fixture
def image_data():
    source_image_path = "resource/source/imgs/外滩人流.png"
    
    with open(source_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    return RetrievalData(items=[
        ImageModalityData(content=base64_image, _id="image_id", metadata={"timestamp": 1234567890})
    ])

def test_text_indexer(mock_processor, text_data):
    text_indexer = TextIndexer(processor=mock_processor)
    text_indexer.index(text_data)

    mock_processor.execute_operation.assert_called()

def test_image_indexer(mock_processor, image_data):
    image_indexer = ImageIndexer(processor=mock_processor)
    image_indexer.index(image_data)

    mock_processor.execute_operation.assert_called()

def test_multimodal_indexer(mock_processor, text_data, image_data):
    multimodal_indexer = MultimodalIndexer(processor=mock_processor)
    combined_data = RetrievalData(items=text_data.items + image_data.items)
    multimodal_indexer.index(combined_data)

    assert mock_processor.execute_operation.call_count == len(combined_data.items)