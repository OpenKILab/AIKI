from bson import ObjectId

from aiki.multimodal import VectorModalityData, VectorHandler, VectorHandlerOP, MultiModalProcessor, ModalityType

from aiki.database.chroma import ChromaDB

import numpy as np

def fake_embedding_func(x, size=2):
    return np.random.rand(size).astype(np.float32)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def test_processor_vector():
    processor = MultiModalProcessor()
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=ChromaDB(), embedding_func=fake_embedding_func))

    _id1 = ObjectId()
    _id2 = ObjectId()
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id1], ["hello"])
    processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.UPSERT, [_id2], ["hello_again"])
    mget_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.MGET, [_id1, _id2])

    assert  len(mget_result) == 2 and isinstance(mget_result[0], VectorModalityData)
    print(mget_result)
    query_result = processor.execute_operation(ModalityType.VECTOR, VectorHandlerOP.QUERY, ["test"])

    print(query_result)
    assert len(query_result[0]) == 2 and isinstance(query_result[0][0], ObjectId)

test_processor_vector()