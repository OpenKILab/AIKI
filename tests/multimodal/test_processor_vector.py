from bson import ObjectId
from sympy import andre

from aiki.multimodal import VectorModalityData, VectorHandler, VectorHandlerOP, MultiModalProcessor, ModalityType

from aiki.database.chroma import ChromaDB

import numpy as np
from unittest.mock import MagicMock

def test_processor_vector():
    processor = MultiModalProcessor()
    mock_embedding_func = MagicMock()
    fake_embeddings = [np.array([1, 0]).astype(np.float32),
     np.array([0, 1]).astype(np.float32),
     np.array([-1, 0]).astype(np.float32),
     np.array([0, -1]).astype(np.float32),
    np.array([0.5, 0.866]).astype(np.float32)]
    mock_embedding_func.side_effect = fake_embeddings
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=ChromaDB(), embedding_func=mock_embedding_func))

    # Test upsert
    _id1 = ObjectId()
    _id2 = ObjectId()
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id1], ["hello"])
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id2], ["hello_again"])
    mget_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.MGET, [_id1, _id2])

    assert  len(mget_result) == 2 and isinstance(mget_result[0], VectorModalityData)

    assert ((mget_result[0].content == fake_embeddings[0]).all()) and ((mget_result[1].content == fake_embeddings[1]).all())
    # Test query

    _id3 = ObjectId()
    _id4 = ObjectId()
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id3], ["hello_again_and_again"])
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id4], ["hello_the_fourth_time"])

    query_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, ["test"])

    # 简单的测试向量余弦相似度
    assert query_result[0][0] == _id2 and query_result[0][1] == _id1 and query_result[0][2] == _id3 and query_result[0][3] == _id4