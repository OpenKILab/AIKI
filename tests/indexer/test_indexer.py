import pytest
from unittest.mock import MagicMock
import logging
from aiki.indexer.indexer import APISummaryGenerator, ImageIndexer, MultimodalIndexer, TextIndexer
from aiki.corpus.mockdatabase import DatabaseConnectionFactory, DatabaseConnection
from aiki.indexer.chunker import FixedSizeChunker
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType

@pytest.fixture
def sourcedb():
    # Create a connection to a JSON file database
    connection = DatabaseConnectionFactory.create_connection('json_file', file_name='test_data.json')
    return connection

@pytest.fixture
def vectordb():
    connection = DatabaseConnectionFactory.create_connection('chroma', index_file='chroma_index')
    return connection

@pytest.fixture
def text_indexer(sourcedb, vectordb):
    return TextIndexer(model_path='path/to/model', sourcedb=sourcedb, vectordb=vectordb)

def test_text_indexer_index(text_indexer, sourcedb, vectordb):
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.TEXT, content="Example text data")
        ]
    )
    
    text_indexer.index(retrieval_data)

    results = vectordb.query(query_texts=["Example text data"], n_results=1)

    assert results['documents'][0] != []
    
@pytest.fixture
def image_indexer(sourcedb, vectordb):
    summary_generator = MagicMock()
    summary_generator.generate_summary.return_value = "Mocked summary for image"
    return ImageIndexer(model_path='path/to/model', sourcedb=sourcedb, vectordb=vectordb, summary_generator=summary_generator)

def test_image_indexer_index(image_indexer, sourcedb, vectordb):
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.IMAGE, content="base64_encoded_image_data")
        ]
    )
    
    image_indexer.index(retrieval_data)

    results = vectordb.query(query_texts=["Mocked summary for image"], n_results=1)
    assert results['documents'][0] != []
    
@pytest.fixture
def multimodal_indexer(sourcedb, vectordb):
    summary_generator = APISummaryGenerator()
    summary_generator.generate_summary = MagicMock(return_value="Mocked summary for image")
    return MultimodalIndexer(model_path='path/to/model', sourcedb=sourcedb, vectordb=vectordb, summary_generator=summary_generator)

def test_multimodal_indexer_index(multimodal_indexer, sourcedb, vectordb):
    retrieval_data = RetrievalData(
        items=[
            RetrievalItem(type=RetrievalType.TEXT, content="Example text data"),
            RetrievalItem(type=RetrievalType.IMAGE, content="base64_encoded_image_data")
        ]
    )
    
    multimodal_indexer.index(retrieval_data)

    # Verify that the text data was indexed
    text_results = vectordb.query(query_texts=["Example text data"], n_results=1)
    assert text_results['documents'][0] != []  # Ensure the query returns results

    # Verify that the image data was indexed
    image_results = vectordb.query(query_texts=["Mocked summary for image"], n_results=1)
    assert image_results['documents'][0] != []  # Ensure the query returns results