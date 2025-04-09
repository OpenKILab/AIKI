import json
import torch
import time
import logging
import requests
import urllib.parse
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluate import (
    run_evaluation, 
    extract_answer
)
from bing_search import (
    serp_google_web_search,
    bing_web_search,
    duckduckgo_web_search,
    extract_relevant_info,
    fetch_page_content,
    extract_snippet_with_context,
)
from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_webpage_to_reasonchain_instructiont_chinese
)

from get_llm import (
    generate_text,
    generate_query,
    generate_summary,
)

# query -> url list -> bing search(like) struct result

top_k = 8
use_jina = False
seq = {}
url_snippets = {}
url_cache = {}
max_doc_len = 3000
max_tokens = 32768

def generate_info(query: str):
    start_time = time.time()  # Start timing the entire function

    # TODO: prev_reasonings -> search_query
    search_query = query
    prev_reasonings = search_query
    model_path = "/fs-computility/ai-shen/shared/hf-hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    BING_SUBSCRIPTION_KEY = "97812f8e5a7d4c2d9b9b5eefd9c7c173"
    if not BING_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")

    bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"

    # Perform the search
    print("Performing Web Search...")
    # results = bing_web_search(search_query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    line1_start = time.time()
    results = serp_google_web_search(search_query, num_web=4)
    line1_end = time.time()
    logging.info(f"Time for serp_google_web_search: {line1_end - line1_start:.4f} seconds")
    # results = duckduckgo_web_search(search_query)
    # TODO: cache
    # search_cache[search_query] = results

    ## bing search(like) result -> relevant_info
    # {
    #                 'id': id + 1,  # Increment id for easier subsequent operations
    #                 'title': result.get('name', ''),
    #                 'url': result.get('url', ''),
    #                 'site_name': result.get('siteName', ''),
    #                 'date': result.get('datePublished', '').split('T')[0],
    #                 'snippet': result.get('snippet', ''),  # Remove HTML tags
    #                 # Add context content to the information
    #                 'context': ''  # Reserved field to be filled later
    #             }
    relevant_info = extract_relevant_info(results)[:top_k]

    seq['relevant_info'] = relevant_info
    # Extract URLs and snippets
    urls_to_fetch = [it['url'] for it in relevant_info]
    # print(f"*************urls_to_fetch:{urls_to_fetch}****************")
    snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

    # TODO: cache
    # Filter URLs that are not cached
    # urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
    # cached_urls = [u for u in urls_to_fetch if u in url_cache]


    # Store info for url_snippets
    for url in urls_to_fetch:
        url_snippets[url] = snippets.get(url, "")
        
    ## url list -> web content
    try:
        print("********fetch page content**********")
        use_jina = True
        print(f"********use jina: {use_jina}**********")
        line2_start = time.time()
        fetched_contents = fetch_page_content(
            list(urls_to_fetch),
            use_jina=use_jina,
            # jina_api_key=jina_api_key,
            # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
        )
        line2_end = time.time()
        logging.info(f"Time for fetch_page_content: {line2_end - line2_start:.4f} seconds")
        print(f"Fetched {len(fetched_contents)} URLs successfully.")
    except Exception as e:
        print(f"Error during batch URL fetching: {e}")
        fetched_contents = {url: f"Error fetching URL: {e}" for url in urls_to_fetch}
        
    # TODO: cache save json
    ## Update cache with fetched contents
    for url, content in fetched_contents.items():
        url_cache[url] = content

    ## web content -> formatted docs
    # After fetching, prepare formatted documents for batch processing
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        raw_context = url_cache.get(url, "")
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
        line3_start = time.time()
        success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
        line3_end = time.time()
        logging.info(f"Time for extract_snippet_with_context: {line3_end - line3_start:.4f} seconds")
        if success:
            context = filtered_context
        else:
            context = raw_context[:max_doc_len*2]

        doc_info['context'] = context
        formatted_documents += f"**Web Page {i + 1}:**\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

    # abstract web content
    ## formatted docs -> web content abstract
        # def generate_webpage_to_reasonchain_batch(
        #     original_questions: List[str],
        #     prev_reasonings: List[str],
        #     search_queries: List[str],
        #     documents: List[str],
        #     dataset_name: str,
        #     batch_output_records: List[Dict],  # New parameter to collect outputs
        #     max_tokens: int = 32768,
        #     coherent: bool = False,
        # )
    user_prompts = [
        get_webpage_to_reasonchain_instructiont_chinese(prev_reasonings, search_query, formatted_documents)
    ]

    prompts = [{"role": "user", "content": up} for up in user_prompts]
    # prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

    # output = llm.generate(
    #     prompts,
    #     sampling_params=SamplingParams(
    #         max_tokens=max_tokens,
    #         temperature=0.7,
    #         top_p=0.8,
    #         top_k=20,
    #         repetition_penalty=1.05,
    #     )
    # )

    # raw_outputs = [out.outputs[0].text for out in output]
    line4_start = time.time()
    raw_outputs = generate_text(prompts)
    line4_end = time.time()
    logging.info(f"Time for generate_text: {line4_end - line4_start:.4f} seconds")
    # extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]
    print("=======raw_outputs========")
    print(raw_outputs)
    print("=======raw_outputs========")
    
    # Extract URLs and names from the results
    webs = [{'url': page['url'], 'name': page['name']} for page in results['webPages']['value']]
    
    end_time = time.time()  # End timing the entire function
    logging.info(f"Total time for generate_info: {end_time - start_time:.4f} seconds")
    logging.info(f"========================================")
    return raw_outputs, webs
    # print("=======extracted_infos========")
    # print(extracted_infos)
    # print("=======extracted_infos========")

def extract_box(input: str):
    boxed_content = re.search(r'{(.*?)}', input)
    if boxed_content:
        boxed_content = boxed_content.group(1)  # Get the content inside the boxed
    else:
        boxed_content = None  # Handle case where no boxed content is found
    return boxed_content
    
def generate_info_simple(query: str):
    start_time_total = time.time()  # 记录总时间开始
    # Start timing for generate_query
    start_time_generate_query = time.time()
    query = generate_query(query)
    
    # Extract the boxed content
    boxed_content = re.search(r'{(.*?)}', query)
    if boxed_content:
        boxed_content = boxed_content.group(1)  # Get the content inside the boxed
    else:
        boxed_content = None  # Handle case where no boxed content is found
    logging.info(query)
    logging.info(boxed_content)
    # End timing for generate_query
    end_time_generate_query = time.time()
    elapsed_time_generate_query = end_time_generate_query - start_time_generate_query
    logging.info(f"Time taken for generate_query: {elapsed_time_generate_query:.4f} seconds")
    
    # Start timing for jina_search
    start_time_jina_search = time.time()
    result = jina_search(boxed_content)
    print("=======jina_res========")
    print(result)
    print("=======jina_res========")
    # End timing for jina_search
    end_time_jina_search = time.time()
    elapsed_time_jina_search = end_time_jina_search - start_time_jina_search
    logging.info(f"Time taken for jina_search: {elapsed_time_jina_search:.4f} seconds")
    
    # 提取相关信息
    info = "**相关信息**\n\n"
    index = 1
    for item in result['data']:
        info += f"{str(index) + '. ' + item['description']}\n\n"
        index += 1
    info = generate_summary(f"问题:{boxed_content} + 相关信息:{info}")
    end_time_jina_search = time.time()
    logging.info("======info========")
    logging.info(info)
    logging.info("======info========")
    # 提取网页链接
    webs = [{'url': item['url'], 'name': item['title']} for item in result['data']]
    logging.info(f"Time taken for jina_search: {elapsed_time_jina_search:.4f} seconds")
    
    end_time_total = time.time()  # 记录总时间结束
    total_elapsed_time = end_time_total - start_time_total
    logging.info(f"Total time taken for generate_info_simple: {total_elapsed_time:.4f} seconds")
    return info, webs

def jina_search(query: str):
    # Construct the URL
    url = "https://s.jina.ai/"
    
    # Set the headers
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer jina_9012aea161ae4a76805f9229459ed51fP89uiwzjfLCOEhfYcgfsp4zzsYiy",
        "X-Respond-With": "no-content"
    }
    
    # Set the data payload
    data = {
        "q": query,
        "gl": "CN",
        "hl": "zh-cn"
    }
    
    # Make the POST request
    response = requests.post(url, headers=headers, json=data)
    
    # Check the response status
    if response.status_code == 200:
        print("Request was successful.")
        # Process the response if needed
        print(response.json())
    else:
        print(f"Request failed with status code: {response.status_code}")
    
    return response.json()

logging.basicConfig(filename='execution_times.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Test the generate_info_simple function with a sample query
    # test_query = "怎么解决数学问题123"
    test_query = """在人类的某次星际探测任务中, 发现一个可能宜居的星球。为了更详细的了解该星球的情况, 需要进行登陆探测。在登陆前, 飞船先环绕该星球在半径为 $$3 R$$ 的圆轨道上做匀速圆周运动, $$R$$ 为该星球的半径。某一时刻, 飞船运行到 $$A$$ 点, 此时飞船将登陆器向反方向射出, 但登陆器仍向前运动, 并进入图示的内部栯圆轨道登上该星球表面, 而飞船立即启动发动机进行调整，使其仍保持在圆轨道运行。登陆器在该星球表面探测一段时间后，重新点火加速沿原来的椭圆轨道回到脱离点实现对接。已知该星球的质量为$M$, 飞船的质量为 $m_{1}$, 登陆器的质量为 $m_{2}$, 且 $m_{1}=2 m_{2}$, 忽略登陆器点火加速的时间,引力常量为 $$G$$ 。求:飞船在反射出登陆器后, 为保持在原轨道运行, 飞船发动机对飞船做的功 $$W$$ ?"""
    test_query = """已知苯甲酸乙酯的沸点为 212.6摄氏度, "乙醚-环已烷-水共沸物" 的沸点为62.1摄氏度。实验室初步分离苯甲酸乙酯、苯甲酸和环已烷的流程如下。下列说法错误的是
A: 操作 a 所使用的主要玻璃仪器为分液漏斗和烧杯
B: 操作b和操作c均为重结晶
C: 无水 $$\{MgSO}_{4}$$ 和饱和碳酸钠溶液的作用相同 
D: 该流程中苯甲酸先转化为苯甲酸钠, 后转化为苯甲酸"""
    # test_query = """where is China?"""
    print(generate_info_simple(test_query))