# test_indexer.py
import pytest
from aiki.indexer.indexer import TextIndexer
from aiki.corpus.database import DatabaseConnectionFactory, DatabaseConnection
from aiki.indexer.chunker import FixedSizeChunker
from aiki.modal.retrieval_data import RetrievalData, RetrievalType

@pytest.fixture
def sourcedb():
    # Create a real connection to an in-memory source database
    connection = DatabaseConnectionFactory.create_connection('in_memory')
    return connection

@pytest.fixture
def vectordb():
    # Create a real connection to a Chroma vector database
    connection = DatabaseConnectionFactory.create_connection('chroma', index_file='chroma_index')
    return connection

@pytest.fixture
def text_indexer(sourcedb, vectordb):
    # Initialize the TextIndexer with real database connections
    return TextIndexer(model_path='path/to/model', sourcedb=sourcedb, vectordb=vectordb)

def test_text_indexer_index(text_indexer, sourcedb):
    # Create a RetrievalData object
    retrieval_data = RetrievalData(
        items=[
            {"type": RetrievalType.TEXT, "content": "Example text data"}
        ]
    )
    
    # Call the index method
    text_indexer.index(retrieval_data)
