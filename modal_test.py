from datetime import datetime
from bson import ObjectId
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData

if __name__ == "__main__":
    a = RetrievalData(
        items=[
            TextModalityData(
                content="abc",
                _id=ObjectId(),
                metadata={
                    "timestamp": datetime.now()
                }
            ),
            ImageModalityData(
                content="abc",
                _id=ObjectId(),
                metadata={
                    "timestamp": datetime.now()
                }
            )
        ]
    )
    print(a.to_json)