from typing import List
from bson import ObjectId
from sympy import andre

from aiki.multimodal import VectorModalityData, VectorHandler, VectorHandlerOP, MultiModalProcessor, ModalityType, \
    BaseModalityData, TextModalityData

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
    def embedding_generator():
        embeddings = [
            np.array([1, 0]).astype(np.float32),
            np.array([0, 1]).astype(np.float32),
            np.array([-1, 0]).astype(np.float32),
            np.array([0, -1]).astype(np.float32),
            np.array([0.5, 0.866]).astype(np.float32)
        ]
        for embedding in embeddings:
            yield [embedding]

    # Create a generator instance
    embedding_gen = embedding_generator()

    # Set the side_effect to use the generator
    mock_embedding_func.side_effect = lambda _: next(embedding_gen)
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=ChromaDB(), embedding_func=mock_embedding_func))

    # Test upsert
    _id1 = ObjectId()
    _id2 = ObjectId()
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=_id1, content="hello")])
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=_id2, content="hello_again")])
    mget_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.MGET, [_id1, _id2])

    assert  len(mget_result) == 2 and isinstance(mget_result[0], VectorModalityData)
    assert (np.squeeze(mget_result[0].content) == fake_embeddings[0]).all() and (np.squeeze(mget_result[1].content) == fake_embeddings[1]).all()

    # Test query

    _id3 = ObjectId()
    _id4 = ObjectId()
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=_id3, content="hello_the_third_time")])
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [TextModalityData(_id=_id4, content="hello_the_fourth_time")])

    query_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, ["test"])

    # 简单的测试向量余弦相似度
    assert query_result[0][0][0] == _id1 
    assert query_result[0][1][0] == _id2 
    assert query_result[0][2][0] == _id4
    assert query_result[0][3][0] == _id3
    
    query_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, query_embeddings=[np.array([1, 0]).astype(np.float32)])
    print(query_result)
    assert query_result[0][0][0] == _id1