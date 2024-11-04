import pytest
import logging
from aiki.indexer.indexer import TextIndexer
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

    assert results['documents'][0] != []  # Ensure the query returns results