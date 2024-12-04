import base64
from sentence_transformers import SentenceTransformer
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.text import TextHandler, TextModalityData
from aiki.multimodal.vector import VectorHandler
from aiki.retriever.retriever import DenseRetriever
from bson import ObjectId

try:
    model = SentenceTransformer('lier007/xiaobu-embedding-v2')
except Exception as e:
    print(f"Error loading model: {e}")
    # Optionally, load a local model or take other actions

embedding_func = model.encode
name = "xiaobu_summary"
processor = MultiModalProcessor()
source_db = JSONFileDB(f"./db/{name}/{name}.json")
chroma_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"./db/{name}/{name}_index")

processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=embedding_func))

dense_retriever = DenseRetriever(processor=processor)

### retrieve
input_data = ""

retrieval_data = RetrievalData(items=[
        TextModalityData(
            content="我上周出去遛狗了么",
            _id=ObjectId(),
            metadata={}
        )
    ])
print("======================")
print("======================")
result = dense_retriever.search(retrieval_data, num=3)
print(result.items)
for item in result.items:
    # image summary
    print(item.metadata.get('summary', ""))
    base64_encoded_image = item.content
    image_data = base64.b64decode(base64_encoded_image)
    with open(f"./tmp/{item._id}.jpg", "wb") as image_file:
        image_file.write(image_data)
        print(item.metadata['summary'])
        
### index
