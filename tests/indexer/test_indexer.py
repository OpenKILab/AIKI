import pytest
import tempfile
import os
from unittest.mock import MagicMock
from aiki.indexer.indexer import APISummaryGenerator, ImageIndexer, MultimodalIndexer, TextIndexer
from aiki.database import BaseKVDatabase, BaseVectorDatabase, JSONFileDB
from aiki.database.chroma import ChromaDB
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType

from chromadb.utils import embedding_functions


@pytest.fixture
def sourcedb():
    # 创建一个临时文件作为 JSON 数据库
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with open(temp_file.name, 'w') as f:
            f.write('{}')
        
        connection = JSONFileDB(temp_file.name)
        yield connection
    finally:
        os.unlink(temp_file.name)  # 确保测试完成后删除临时文件

@pytest.fixture
def vectordb():
    connection = ChromaDB(collection_name="text_index", persist_directory="./aiki/corpus/db/test_index")
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
    embedding_func=embedding_functions.DefaultEmbeddingFunction()
    embedding = embedding_func(["Example text data"])[0]
    results = vectordb.query(query_embeddings=embedding, top_k=1)
    print(results)
    assert results != []

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
    
    mock_embedding = [0.0] * 384

    image_indexer.index(retrieval_data)

    results = vectordb.query(query_embeddings=[mock_embedding], top_k=1)

    assert results[0] != []
    
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
    
    mock_text_embedding = [0.0] * 384  
    mock_image_embedding = [0.0] * 384  

    multimodal_indexer.index(retrieval_data)

    text_results = vectordb.query(query_embeddings=[mock_text_embedding], top_k=1)
    assert text_results[0] != []

    image_results = vectordb.query(query_embeddings=[mock_image_embedding], top_k=1)
    assert image_results[0] != []