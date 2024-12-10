from aiki.externalsearch.external_search import ExternalSearch
from aiki.modal.retrieval_data import TextModalityData, ImageModalityData, RetrievalData
from dotenv import load_dotenv
import os
from bson import ObjectId

load_dotenv()

API_KEY = os.getenv("API_KEY")
# KEY = "serpapi_key" # replace with ur serpapi key

external_search = ExternalSearch(API_KEY)
inputs = {
    "text": "question",
    "image_url": "replace with ur image_url"
}

# 创建原始的retrieval_data
query = RetrievalData(items=[
    TextModalityData(
        _id=ObjectId(),
        content="question",
    ),
    ImageModalityData(
        _id=ObjectId(),
        content="replace with ur image_url",
    )
])


# 在调用search_multimodal时，将query传入，这样搜索后会自动填充query的metadata
results = external_search.search_multimodal(inputs, retrieval_data=query)

# 检查query中的metadata
for idx, item in enumerate(query.items, start=1):
    print(f"Item {idx}: {type(item).__name__}, metadata={item.metadata}")

