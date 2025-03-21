import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from duckduckgo_search import DDGS
from aiki.externalsearch.external_search import ExternalSearch

import logging
# logging.basicConfig(filename='bing_search.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)



def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key="jina_e600ddc05f8641b780644d16a65e97a3EmrQNxyz_kVWjePs7lFBKF4m1NIy", snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    jina_api_key="jina_e600ddc05f8641b780644d16a65e97a3EmrQNxyz_kVWjePs7lFBKF4m1NIy"
    try:
        if use_jina:
            # logging.info("*********extract with jina************")  # Log instead of print
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            # logging.info("*********extract with BeautifulSoup************")  # Log instead of print            
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=4, use_jina=True, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns None or raises an exception if the request times out.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        if response.status_code == 401:
            print("认证失败：API key 无效或已过期。请检查您的 Bing Search API key 是否正确。")
            return {}
        response.raise_for_status()
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Bing Web Search 请求超时 ({timeout} 秒)，查询内容: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Bing Web Search 请求出错: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")
        return {}

def convert_duckduckgo_to_bing(duckduckgo_results, search_query):
    bing_results = {
        "_type": "SearchResponse",
        "queryContext": {
            "originalQuery": search_query
        },
        "webPages": {
            "webSearchUrl": f"https://www.bing.com/search?q={search_query.replace(' ', '+')}",
            "totalEstimatedMatches": len(duckduckgo_results),
            "value": []
        }
    }

    for result in duckduckgo_results:
        bing_result = {
            "id": f"https://api.bing.microsoft.com/api/v7/#WebPages.{len(bing_results['webPages']['value'])}",
            "name": result["title"],
            "url": result["href"],
            "isFamilyFriendly": True,
            "displayUrl": result["href"],
            "snippet": result["body"],
            "dateLastCrawled": "2025-02-20T05:17:00.0000000Z",  # 示例日期
            "language": "en",
            "isNavigational": False,
            "noCache": False,
            "siteName": result["href"].split('/')[2]  # 从 URL 中提取站点名称
        }
        bing_results["webPages"]["value"].append(bing_result)

    return bing_results


def convert_search_results(query, input_data):
    # 解析输入数据
    results = []
    for item in input_data:
        result = {
            'id': f"https://api.bing.microsoft.com/api/v7/#WebPages.{item['position'] - 1}",
            'name': item['title'],
            'url': item['link'],
            'isFamilyFriendly': True,
            'displayUrl': item['displayed_link'],
            'snippet': item['snippet'],
            'dateLastCrawled': '2025-02-20T05:17:00.0000000Z',
            'language': 'en',
            'isNavigational': False,
            'noCache': False,
            'siteName': item['source']
        }
        results.append(result)

    # 构建输出格式
    output = {
        '_type': 'SearchResponse',
        'queryContext': {
            'originalQuery': query
        },
        'webPages': {
            'webSearchUrl': 'https://www.bing.com/search?q='+query,
            'totalEstimatedMatches': len(results),
            'value': results
        }
    }
    
    return output

def duckduckgo_web_search(query):
    import time
    # time.sleep(5.)
    ddgs = DDGS(timeout=60)
    results = ddgs.text(query, max_results=10, backend='auto')
    results = convert_duckduckgo_to_bing(results, query)
    return results
    
def serp_google_web_search(query):
    es = ExternalSearch(api_key="13712b498314b1bc54738124b9cda744ccf5e004a27635fbf13350fe1a50c652")
    res = es.search_text(query, num_results = 10)
    output_data = convert_search_results(query, res)
    return output_data

def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info

if __name__ == "__main__":
    print(serp_google_web_search("火龙果"))
# # ------------------------------------------------------------

# if __name__ == "__main__":
#     # Example usage
#     # Define the query to search
#     query = "火龙果"
    
#     # Subscription key and endpoint for Bing Search API
#     BING_SUBSCRIPTION_KEY = "97812f8e5a7d4c2d9b9b5eefd9c7c173"
#     if not BING_SUBSCRIPTION_KEY:
#         raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
#     bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
#     # Perform the search
#     print("Performing Bing Web Search...")
#     search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    
#     print("Extracting relevant information from search results...")
#     extracted_info = extract_relevant_info(search_results)

#     print("Fetching and extracting context for each snippet...")
#     for info in tqdm(extracted_info, desc="Processing Snippets"):
#         full_text = extract_text_from_url(info['url'], use_jina=False)  # Get full webpage text
#         if full_text and not full_text.startswith("Error"):
#             success, context = extract_snippet_with_context(full_text, info['snippet'])
#             if success:
#                 info['context'] = context
#             else:
#                 info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
#         else:
#             info['context'] = f"Failed to fetch full text: {full_text}"

#     print("Your Search Query:", query)
#     print("Final extracted information with context:")
#     print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
