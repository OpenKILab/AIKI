# test_indexer.py
import pytest
from aiki.indexer.indexer import TextIndexer
from aiki.corpus.mockdatabase import DatabaseConnectionFactory, DatabaseConnection
from aiki.indexer.chunker import FixedSizeChunker
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType

@pytest.fixture
def sourcedb():
    connection = DatabaseConnectionFactory.create_connection('in_memory')
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
    
    results = vectordb.query(query_texts=["Example text data"], n_results=1)
    assert len(results) > 0  # 确保查询结果不为空