from aiki.externalsearch.external_search import ExternalSearch
from aiki.modal.retrieval_data import TextModalityData, ImageModalityData, RetrievalData

from bson import ObjectId

API_KEY = "e98317cdb3df846c1cdf19e1617c09e56842ea738f4f86e3b09b1c1d5ef054e2"

external_search = ExternalSearch(API_KEY)
inputs = {
    "text": "In what year did humans first land on this planet?",
    "image_url": "https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_2.jpeg"
}

# 创建原始的retrieval_data
query = RetrievalData(items=[
    TextModalityData(
        _id=ObjectId(),
        content="In what year did humans first land on this planet?",
    ),
    ImageModalityData(
        _id=ObjectId(),
        content="https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_2.jpeg",
    )
])


# 在调用search_multimodal时，将query传入，这样搜索后会自动填充query的metadata
results = external_search.search_multimodal(inputs, retrieval_data=query)

# 检查query中的metadata
for idx, item in enumerate(query.items, start=1):
    print(f"Item {idx}: {type(item).__name__}, metadata={item.metadata}")

