from aiki.modal.retrieval_data import TextModalityData, ImageModalityData, RetrievalData
from typing import Dict, Any
from serpapi import GoogleSearch
import time

class ExternalSearch:
    def __init__(self, api_key: str, retry_attempts: int = 3):
        """
        初始化 OmniSearch 模块。
        :param api_key: SerpAPI 的 API 密钥，用于授权搜索请求。
        :param retry_attempts: 搜索失败时的最大重试次数。

        pip install serpapi
        """
        self.api_key = api_key
        self.retry_attempts = retry_attempts

    # 统一搜索接口
    def search(self, query: str, search_type: str) -> list:
        """
        统一搜索接口，支持文本搜索、图片搜索、反向图片搜索。
        
        :param query: 搜索内容，可以是文本或图片 URL。
        :param search_type: 搜索类型：
            - 'text': 文本搜索。
            - 'image_text': 基于文本的图片搜索。
            - 'image_url': 反向图片搜索。
        :return: 搜索结果的列表。
        """
        if search_type == "text":
            raw_results = self.search_text(query)
            return self._format_text_results(raw_results)

        elif search_type == "image_text":
            raw_results = self.search_image_by_text(query)
            return self._format_image_results(raw_results, mode="image_text")

        elif search_type == "image_url":
            raw_results = self.search_image_by_url(query)
            return self._format_image_results(raw_results, mode="image_url")

        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    def search_text(self, query: str, num_results: int = 5) -> list:
        """
        执行文本搜索。
        
        :param query: 查询文本。
        :param num_results: 返回结果的最大数量。
        :return: 包含文本搜索结果的列表。
        """
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        return self._execute_search(params, result_key="organic_results")

    def search_image_by_text(self, query: str) -> list:
        """
        执行基于文本的图片搜索。
        
        :param query: 查询文本。
        :return: 包含图片搜索结果的列表。
        """
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.api_key
        }
        return self._execute_search(params, result_key="images_results")

    def search_image_by_url(self, image_url: str) -> dict:
        """
        执行反向图片搜索。
        
        :param image_url: 图片的 URL。
        :return: 搜索结果的字典。
        """
        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "api_key": self.api_key
        }
        return self._execute_search(params)

    def _execute_search(self, params: dict, result_key: str = None) -> list:
        """
        执行搜索请求，处理 API 返回结果。
        
        :param params: 搜索请求的参数字典。
        :param result_key: 返回结果中的关键字段，如果指定，则提取该字段的内容。
        :return: 搜索结果的列表或字典。
        """
        for attempt in range(self.retry_attempts):
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                if result_key and result_key in results:
                    return results[result_key]
                return results
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2)  # 等待2秒后重试
        print("All retries failed.")
        return []

    def _format_text_results(self, results: list) -> list:
        """
        格式化文本搜索结果为列表。
        
        :param results: 原始文本搜索结果。
        :return: 格式化后的列表，每个元素包含文本搜索的标题、摘要、链接。
        """
        formatted_results = []
        for item in results:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            formatted_results.append({
                "type": "text",
                "title": title,
                "snippet": snippet,
                "link": link
            })
        return formatted_results

    def _format_image_results(self, results, mode: str) -> list:
        """
        格式化图片搜索结果为可供系统使用的列表。
        
        :param results: 原始图片搜索结果（可能是列表或字典）。
        :param mode: 表示当前处理的搜索模式:
            - "image_text": 来自文本的图片搜索 (google_images)
            - "image_url": 反向图片搜索 (google_reverse_image)
        :return: 格式化后的图片信息列表。
        """

        formatted_results = []

        if mode == "image_text":
            # results 应该是 images_results 列表
            if isinstance(results, list):
                for item in results:
                    title = item.get("title", "")
                    image_url = item.get("thumbnail", "")
                    link = item.get("link", "")
                    source = item.get("source", "")
                    formatted_results.append({
                        "type": "image",
                        "source_type": "image_text",
                        "title": title,
                        "image_url": image_url,
                        "original_link": link,
                        "source": source
                    })
            return formatted_results

        elif mode == "image_url":
            # 反向图片搜索结果可能包含 knowledge_graph 或 image_results
            if isinstance(results, dict):
                # 若有 knowledge_graph
                if "knowledge_graph" in results:
                    kg = results["knowledge_graph"]
                    title = kg.get("title", "")
                    description = kg.get("description", "")
                    header_images = kg.get("header_images", [])

                    for img_item in header_images:
                        image_url = img_item.get("source", "")
                        formatted_results.append({
                            "type": "image",
                            "source_type": "image_url_kg",
                            "title": title,
                            "description": description,
                            "image_url": image_url
                        })

                # 若无 knowledge_graph，则尝试 image_results
                if not formatted_results and "image_results" in results:
                    for item in results["image_results"]:
                        snippet = item.get("snippet", "")
                        link = item.get("link", "")
                        image_url = item.get("image", {}).get("thumbnail", "")
                        if not image_url:
                            image_url = item.get("thumbnail", "")

                        formatted_results.append({
                            "type": "image",
                            "source_type": "image_url_ir",
                            "title": snippet,
                            "image_url": image_url,
                            "original_link": link
                        })

            return formatted_results

        return formatted_results

    def search_multimodal(self, inputs: dict,retrieval_data: RetrievalData = None) -> dict:
        """
        多模态搜索接口，可同时处理文本和图片输入。
        
        假设输入格式为：
        {
            "text": "some query text",
            "image_url": "http://example.com/image.jpg"
        }
        
        我们将基于text执行文本搜索和图片搜索（image_text），基于image_url执行反向图片搜索，然后将结果融合。
        
        :param inputs: 包含文本及图片URL的字典
        :return: 包含多模态结果的字典。
        """
        
        text_query = inputs.get("text", "")
        image_url_query = inputs.get("image_url", "")

        # 根据需求灵活修改下列搜索逻辑
        text_results = self.search(text_query, "text") if text_query else []
        image_text_results = self.search(text_query, "image_text") if text_query else []
        image_url_results = self.search(image_url_query, "image_url") if image_url_query else []

        # 简单的融合策略：直接将3种结果都返回
        results =  {
            "text_results": text_results,
            "image_text_results": image_text_results,
            "image_url_results": image_url_results
        }

        if retrieval_data is not None:
            self.populate_retrieval_data_with_omni_results(retrieval_data, results)

        return results

    def populate_retrieval_data_with_omni_results(self, retrieval_data: RetrievalData, external_search_results: Dict[str, Any]) -> None:
        """
        将Omnisearch的结果填充回RetrievalData中对应的模态数据项里。

        :param retrieval_data: 一个 RetrievalData 实例，其中的 items 包含 TextModalityData、ImageModalityData 等。
        :param omni_results: Omnisearch搜索出来的结果字典，格式如下：
            {
                "text_results": [...],
                "image_text_results": [...],
                "image_url_results": [...]
            }
        """
        text_results = external_search_results.get("text_results", [])
        image_text_results = external_search_results.get("image_text_results", [])
        image_url_results = external_search_results.get("image_url_results", [])

        for item in retrieval_data.items:
            # 确保metadata不为空
            if item.metadata is None:
                item.metadata = {}

            # 根据item的类型，将相应的结果放入metadata
            if isinstance(item, TextModalityData):
            # 在文本模态数据的metadata中存text_results
                item.metadata["text_results"] = text_results
            elif isinstance(item, ImageModalityData):
                # 在图像模态数据的metadata中存image_text_results和image_url_results
                item.metadata["image_text_results"] = image_text_results
                item.metadata["image_url_results"] = image_url_results

