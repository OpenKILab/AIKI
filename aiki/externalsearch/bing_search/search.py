import json
import torch

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
    generate_text
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
    results = serp_google_web_search(search_query)
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
        fetched_contents = fetch_page_content(
            list(urls_to_fetch),
            use_jina=use_jina,
            # jina_api_key=jina_api_key,
            # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
        )
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
        success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
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
    raw_outputs = generate_text(prompts)
    # extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]
    print("=======raw_outputs========")
    print(raw_outputs)
    print("=======raw_outputs========")
    
    # Extract URLs and names from the results
    webs = [{'url': page['url'], 'name': page['name']} for page in results['webPages']['value']]
    
    return raw_outputs, webs
    # print("=======extracted_infos========")
    # print(extracted_infos)
    # print("=======extracted_infos========")