from typing import List, Dict, Any

class BaseReranker:
    def __init__(self, config):
        self.config = config
        
    def rerank(self, query_list: List[Dict[str, Any]], data_list: List[List[Dict[str, Any]]], topk: int = None) -> List[List[Dict[str, Any]]]:
        # query_list example:
        # [
        #     {
        #         "text": "What is the capital of France?",
        #         "image": image_data,  # Optional
        #         "audio": audio_data   # Optional
        #     },
        #     ...
        # ]

        # data_list example:
        # [
        #     [
        #         {
        #             "document_id": "doc1",
        #             "content": "Paris is the capital of France.",
        #             "score": 0.8
        #         },
        #         ...
        #     ],
        #     ...
        # ]
        ...
