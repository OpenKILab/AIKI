# 新的导入语句
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info, fetch_image
from transformers import Qwen2_5_VLForConditionalGeneration
import torch.nn as nn
import logging
import uuid
import argparse
import time
import os
import asyncio
import aiohttp
from contextlib import asynccontextmanager
from openai import AsyncOpenAI, OpenAI
from openai._types import NOT_GIVEN
import base64
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import aiohttp
import asyncio
import json
from typing import List, Dict, Any
from types import SimpleNamespace
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template for the prompt
PROMPT_TEMPLATE = """{question}"""

# Create Flask app
app = FastAPI(title="Value Guidance API", description="API for value guidance and inference", version="1.0.0")


# Configuration - moved to a central location
CONFIG = {
    "model_endpoints": {
        "safety": "http://172.30.56.0:5001/v1",
        "value": "http://172.30.56.0:5002/v1",
        "knowledge": "http://172.30.56.0:5003/v1",
        "gating": f"http://172.30.52.0:7001/predict",
        "rag": "http://172.30.8.157:10033/info/",
        "inference": {
            "Qwen2.5VL-7B-SFT": "http://172.30.56.0:8001/v1",
            # "Qwen2.5VL-7B-RL": "http://172.30.56.0:8003/v1",
            "Qwen2.5VL-7B-RL": "http://0.0.0.0:10022/v1",
            "DeepSeek": "https://ark.cn-beijing.volces.com/api/v3",
            "InternVL2.5-78B": "https://chat.intern-ai.org.cn/api/v1",
        },
        "served_model_name": {
            "Qwen2.5VL-7B-SFT": "inference",
            "Qwen2.5VL-7B-RL": "inference",
            "DeepSeek": "ep-20250217120545-tvgfm",
            "InternVL2.5-78B": "internvl2.5-latest",
        },
        "default_inference_model_name": "Qwen2.5VL-7B-RL",
    },
    # "tokenizer_path": "/fs-computility/ai-shen/mllm_safety-shared/projects/trustworthy-infer/VisVM/value_train/train/output_safety/checkpoint-1048",
    # "processor_path": "/fs-computility/ai-shen/mllm_safety-shared/projects/trustworthy-infer/VisVM/value_train/train/output_safety/checkpoint-1048",

    # "tokenizer_path": "/fs-computility/ai-shen/mllm_safety-shared/projects/trustworthy-infer/inference_model/grop0324",
    # "processor_path": "/fs-computility/ai-shen/mllm_safety-shared/projects/trustworthy-infer/inference_model/grop0324",
    "tokenizer_path": "/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "processor_path": "/fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "default_params": {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 2048,
        "chunk_L": 100,
        "num_beams": 1,
        "num_candidates": 4,
        "max_chunks": 20
    }
}


class GenerateRequest(BaseModel):
    """Request model for generating responses"""
    # question, message, prompt: one of them must be provided
    question: Optional[str] = None
    message: Optional[str] = None
    prompt: Optional[str] = None 
    # only for step generate
    generated_response: Optional[str] = ""
    step : Optional[int] = 0
    # image, mediaprompt, image_url : one of them must be provided
    image: Optional[str] = ""
    mediaprompt: Optional[str] = "[]"
    image_url: Optional[str] = ""
    # beam_search_params and sampling_params are optional
    temperature: Optional[float] = CONFIG["default_params"]["temperature"]
    top_p: Optional[float] = CONFIG["default_params"]["top_p"]
    top_k: Optional[int] = CONFIG["default_params"]["top_k"]
    max_tokens: Optional[int] = CONFIG["default_params"]["max_tokens"]
    chunk_L: Optional[int] = CONFIG["default_params"]["chunk_L"]
    num_beams: Optional[int] = CONFIG["default_params"]["num_beams"]
    num_candidates: Optional[int] = CONFIG["default_params"]["num_candidates"]
    max_chunks: Optional[int] = CONFIG["default_params"]["max_chunks"]
    # inference model name, choice one from the model_endpoints
    inference_model_name: Optional[str] = CONFIG["model_endpoints"]["default_inference_model_name"]
    # score setting for value model, default is {"knowledge": 1, "safety": 1, "value": 1}
    setting: Optional[Dict[str, float]] = {"knowledge": 1, "safety": 1, "value": 1}
    # whether to use rag
    enable_rag: Optional[bool] = True
    enable_gating: bool = Field(False, description="Enable gating for generation")


# Global model instances
model = None
value_model = None

import aiohttp

# Add this function to fetch RAG information
import requests
from requests.exceptions import RequestException

async def fetch_rag_info(question: str, image_url: str = None) -> Dict[str, List[str]]:
    """Fetch relevant information from RAG API asynchronously"""
    result = {"info": "", "urls": []}
    rag_url = CONFIG["model_endpoints"]["rag"]
    payload = {"question": question, "image": ""}
    
    if image_url:
        payload["image"] = image_url

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rag_url, json=payload, timeout=20) as response:
                if response.status == 200:
                    data = await response.json()
                    print("rag info:", data)
                    result["info"] = data.get("info", "")
                    result["urls"] = data.get("webs", [])
                else:
                    logger.warning(f"RAG API returned status {response.status}")
                    
    except asyncio.TimeoutError:
        logger.warning("RAG API request timed out")
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching RAG info: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
    return result


async def fetch_gating_info(question: str, image_url: str = None) -> Dict:

    setting = {"safety": 1.0, "value": 1.0, "knowledge": 1.0}
    endpoint = CONFIG["model_endpoints"]["gating"] 
    payload = {"question": question, "image": None}

    if image_url:
        payload["image"] = image_url

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, timeout=20) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Get gating info: {}".format(data))
                    if image_url and data.get("get_image", False):
                        logger.error("Gating server dont get image!")
                    return_setting = data.get("setting", {})
                    setting["safety"] = return_setting.get("safe",  -10)
                    setting["value"]  = return_setting.get("value", -10)
                    setting["knowledge"] = max(return_setting.get("knowledge", -10), return_setting.get("general", -10), return_setting.get("math", -10) ,-10)
                else:
                    logger.warning(f"RAG API returned status {response.status}")
                    
    except asyncio.TimeoutError:
        logger.warning("RAG API request timed out")
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching RAG info: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
    return setting



class RequestPostAsyncInferenceModel:
    """Asynchronous wrapper for the inference model using requests.post"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])
        self.processor = AutoProcessor.from_pretrained(
            CONFIG["processor_path"], 
            min_pixels=256*28*28, 
            max_pixels=512*28*28
        )
            
    async def generate_internvl2_5(self, llm_inputs: List[Dict], sampling_params: Any, inference_name: str = "InternVL2.5-78B", final_think: bool = False) -> List[Any]:
        """Async generation of responses using requests.post"""
        async with aiohttp.ClientSession() as session:

            assert inference_name in ["InternVL2.5-78B"], f"We only support internvl2.5 model, but got {inference_name}"
        
            endpoint = CONFIG["model_endpoints"]["inference"][inference_name] + "/chat/completions"
            internvl_api_key = os.getenv("INTERNVL_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {internvl_api_key}"
            }
            stop_sequences = ["</think>", "<|im_end|>", "<|endoftext|>", "\n\n"] if not final_think else ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]

            tasks = []
            for llm_input in llm_inputs:
                messages = llm_input["messages"]
                generated_response = llm_input["generated_response"]
                messages = [
                    messages[0],
                    {
                        "role": "assistant",
                        "content": generated_response
                    }
                ]
                
                payload = {
                    "model": "internvl2.5-latest",
                    "messages": messages,
                    "max_tokens": sampling_params.max_tokens,
                    # "repetition_penalty": 1.1,
                    "temperature": sampling_params.temperature,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "stop": stop_sequences,
                    "n": sampling_params.n,
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
                # logger.info("Sending request to internvl2.5 model with url: {}, payload: {}".format(endpoint, payload))
                task = session.post(
                    url=endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    # timeout=50,
                )
                tasks.append(task)
            
            try:
                responses = await asyncio.gather(*tasks)
                outputs = []
                for response in responses:
                    if response.status == 200:
                        outputs.append(await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                assert len(llm_inputs) == 1, "internvl model only support num_beam = 1"

                for output in outputs: # internvl support sampling_parameters.n
                    for i in range(len(output["choices"])):
                        choice = output["choices"][i]
                        content = choice["message"]["content"]
                        stop_tokens = ["</think>", "<|im_end|>", "<|endoftext|>", "\n\n"] if not final_think else ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]
                        
                        for stop in stop_tokens:
                            if stop in content:
                                content = stop.join(content.split(stop)[:-1])
                                output["choices"][i]["message"]["content"] = content
                                output['choices'][i]['finish_reason'] = "stop"
                                output['choices'][i]['stop_reason'] = stop
                                break

                outputs= [json.loads(json.dumps(item), object_hook=lambda d: SimpleNamespace(**d)) for item in outputs]
                logger.info(f"Got response from internvl2.5 model: {outputs}")
                return outputs
            except Exception as e:
                logger.error(f"Error in async generation: {str(e)}", exc_info=True)
                raise

    async def generate_deepseek(self, llm_inputs: List[Dict], sampling_params: Any, inference_name: str = "deepseek", final_think: bool = False) -> List[Any]:
        """Async generation of responses using requests.post"""
        async with aiohttp.ClientSession() as session:

            assert inference_name in ["DeepSeek"], f"We only support deepseek model, but got {inference_name}"

            endpoint = CONFIG["model_endpoints"]["inference"][inference_name] + "/chat/completions"
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {deepseek_api_key}"
            }
            stop_sequences = ["</think>", "<|im_end|>", "<|endoftext|>", "\n\n"] if not final_think else ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]

            tasks = []
            for llm_input in llm_inputs:
                messages = llm_input["messages"]
                generated_response = llm_input["generated_response"]
                messages = [
                    messages[0],
                    {
                        "role": "assistant",
                        "content": generated_response
                    }
                ]
                
                payload = {
                    "model": "ep-20250217120545-tvgfm",
                    "messages": messages,
                    "max_tokens": sampling_params.max_tokens,
                    # "repetition_penalty": 1.1,
                    "temperature": sampling_params.temperature,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "stop": stop_sequences,
                    "n": 1, # deepseek model only support n=1
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                }
                
                for i in range(sampling_params.n):
                    logger.info("Sending request to deepseek model with url: {}, payload: {}".format(endpoint, payload))
                    task = session.post(
                        url=endpoint,
                        headers=headers,
                        data=json.dumps(payload),
                        # timeout=50,
                    )
                    tasks.append(task)
            
            try:
                responses = await asyncio.gather(*tasks)
                outputs = []
                for response in responses:
                    if response.status == 200:
                        outputs.append(await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")

                # 这里将每个output的choices数组合并，以模拟这里为n>1的情况
                assert len(llm_inputs) == 1, "deepseek model only support num_beam = 1"
                for output in outputs:
                    
                    choice = output["choices"][0]
                    content = choice["message"]["content"]
                    stop_tokens = ["</think>", "<|im_end|>", "<|endoftext|>", "\n\n"] if not final_think else ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]

                    for stop in stop_tokens:
                        if stop in content:
                            content = stop.join(content.split(stop)[:-1])
                            output["choices"][0]["message"]["content"] = content
                            output['choices'][0]['finish_reason'] = "stop"
                            output['choices'][0]['stop_reason'] = stop
                            break
                for output in outputs[1:]:
                    outputs[0]["choices"] += output["choices"]

                outputs = [outputs[0],]

                # 使用SampleNameSpace来转化response
                outputs= [json.loads(json.dumps(item), object_hook=lambda d: SimpleNamespace(**d)) for item in outputs]
                return outputs
            except Exception as e:
                logger.error(f"Error in async generation: {str(e)}", exc_info=True)
                raise Exception(f"Error in async generation: {str(e)}")
            finally:
                # Don't close session here as it might be reused
                pass
        


    async def generate(self, llm_inputs: List[Dict], sampling_params: Any, inference_name: str = "sft", final_think: bool = False) -> List[Any]:
        if inference_name == "DeepSeek":
            return await self.generate_deepseek(llm_inputs, sampling_params, inference_name, final_think)
        if inference_name == "InternVL2.5-78B":
            return await self.generate_internvl2_5(llm_inputs, sampling_params, inference_name, final_think)

        """Async generation of responses using requests.post"""
        async with aiohttp.ClientSession() as session:

            endpoint = CONFIG["model_endpoints"]["inference"][inference_name] + "/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-123456"
            }
            
            stop_sequences = ["</think>", "<|im_end|>", "<|endoftext|>", "\n\n"] if not final_think else ["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]
            
            tasks = []
            for llm_input in llm_inputs:
                messages = llm_input["messages"]
                generated_response = llm_input["generated_response"]
                messages = [
                    messages[0],
                    {
                        "role": "assistant",
                        "content": generated_response,
                    }
                ]
                
                payload = {
                    "model": "Qwen2.5-VL-7B-Instruct",
                    "messages": messages,
                    "max_tokens": sampling_params.max_tokens,
                    # "repetition_penalty": 1.1,
                    # "presence_penalty": 1.0,
                    # "frequency_penalty": 0.1,
                    "temperature": sampling_params.temperature,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "stop": stop_sequences,
                    "n": sampling_params.n,
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                    # "skip_special_tokens": False,
                }
                
                task = session.post(
                    url=endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    # timeout=50,
                )
                tasks.append(task)
            
            try:
                responses = await asyncio.gather(*tasks)
                outputs = []
                for response in responses:
                    if response.status == 200:
                        outputs.append(await response.json())
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                # 使用SampleNameSpace来转化response
                outputs= [json.loads(json.dumps(item), object_hook=lambda d: SimpleNamespace(**d)) for item in outputs]
                return outputs
            except Exception as e:
                logger.error(f"Error in async generation: {str(e)}", exc_info=True)
                raise
            finally:
                # Don't close session here as it might be reused
                pass




class AsyncValueModel:
    """Asynchronous wrapper for the value models (safety, value, knowledge)"""
    
    def __init__(self):
        # Create async clients for each value model
        self.safety_client = AsyncOpenAI(
            api_key='empty',
            base_url=CONFIG["model_endpoints"]["safety"]
        )
        self.value_client = AsyncOpenAI(
            api_key='empty',
            base_url=CONFIG["model_endpoints"]["value"]
        )
        self.knowledge_client = AsyncOpenAI(
            api_key='empty',
            base_url=CONFIG["model_endpoints"]["knowledge"]
        )
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])
        self.processor = AutoProcessor.from_pretrained(CONFIG["processor_path"])
        
        # Cache token IDs
        self.yes_token_id = self.tokenizer.encode("yes")[0]
        self.no_token_id = self.tokenizer.encode("no")[0]
    
    async def evaluate_candidate(self, messages, generated_text, setting=None):
        """Evaluate a single candidate with all three value models in parallel"""
        
        if "<reference_info>" in generated_text and "</reference_info>" in generated_text:
            # delete all the content between <reference_info> and </reference_info>
            logger.info("Delete all the content between <reference_info> and </reference_info>")
            generated_text = re.sub(r'<reference_info>.*?</reference_info>', '', generated_text, flags=re.DOTALL)

        current_messages = [
            messages[0],
            {
                "role": "assistant",
                "content": generated_text
            }
        ]
        
        # Common parameters for all evaluation calls
        eval_params = {
            "max_tokens": 1,
            "temperature": 1,
            "logprobs": True,
            "top_logprobs": 2,
            "logit_bias": {
                self.yes_token_id: 20,
                self.no_token_id: 20,
            },
        }
        
        # Create tasks for parallel execution
        safety_task = self.safety_client.chat.completions.create(
            model="safety",
            messages=current_messages,
            **eval_params
        )
        
        value_task = self.value_client.chat.completions.create(
            model="value",
            messages=current_messages,
            **eval_params
        )
        
        knowledge_task = self.knowledge_client.chat.completions.create(
            model="knowledge",
            messages=current_messages,
            **eval_params
        )
        
        try:
            # Execute all evaluation tasks in parallel
            safety_resp, value_resp, knowledge_resp = await asyncio.gather(
                safety_task, value_task, knowledge_task
            )
            
            # Extract scores from responses
            safety_score = self._get_score(safety_resp)
            value_score = self._get_score(value_resp)
            knowledge_score = self._get_score(knowledge_resp)
            
            try:
                score = setting["safety"] * safety_score + setting["value"] * value_score + setting["knowledge"] * knowledge_score
            except:
                score = safety_score + value_score + knowledge_score
            score = round(score, 3)

            return {
                "safety": safety_score,
                "value": value_score,
                "knowledge": knowledge_score,
                "score": score
            }
        except Exception as e:
            logger.error(f"Error in async evaluation: {str(e)}", exc_info=True)
            # Return a default negative score on error
            return {
                "safety": -10,
                "value": -10,
                "knowledge": -10,
                "score": -30
            }
    
    def _get_score(self, response):
        """Extract score from model response"""
        try:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            top_logprobs_dict = {d.token: d for d in top_logprobs}
            score = top_logprobs_dict["yes"].logprob - top_logprobs_dict["no"].logprob
            sigmoid_score = round(1 / (1 + np.exp(-score)), 3)
            return sigmoid_score
            # return top_logprobs_dict["yes"].logprob - top_logprobs_dict["no"].logprob
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Error extracting score: {str(e)}")
            return -10  # Return negative score on error
    
    async def get_reward(self, messages, generated_texts, setting=None):
        """Get rewards for multiple candidates in parallel"""
        if not generated_texts:
            return []
        
        # Evaluate all candidates in parallel
        tasks = [self.evaluate_candidate(messages, text, setting) for text in generated_texts]
        try:
            return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}", exc_info=True)
            # Return default negative scores on error
            return [{"safety": -10, "value": -10, "knowledge": -10, "score": -30} 
                   for _ in range(len(generated_texts))]


def load_models():
    """Load all necessary models"""
    global model, value_model
    
    logger.info("Loading inference model...")
    model = RequestPostAsyncInferenceModel()
    
    logger.info("Loading value models...")
    value_model = AsyncValueModel()
    
    logger.info("All models loaded successfully.")

def pil_to_base64(image):
    """Convert PIL image to base64 string"""
    if image is None:
        return None
        
    try:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_string}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None


async def value_guidance_search_inference_stream(question, image, media_info, beam_search_params, sampling_params, inference_name="sft", setting=None, enable_gating=False, enable_rag=True):
    """Generator function for streaming value guidance search results with async support"""
    logger.info(f"Starting value guidance search for question: {question}")
    logger.info(f"\033[31mGet Image!\033[0m" if image else "\033[32mNo Image!\033[0m")
    logger.info(f"beam search:{beam_search_params}\nsampling_params:{sampling_params}")
    try:
        # RAG 
        image_url = pil_to_base64(image) if image else None

        rag_result = {"info": "", "urls": []}
        if enable_rag:
            rag_result = await fetch_rag_info(question, image_url)
        rag_info = rag_result["info"]
        source_urls = rag_result["urls"]

        if setting == None and enable_gating:
            setting = await fetch_gating_info(question, image_url)

        # Prepare input
        content = PROMPT_TEMPLATE.format(question=question)
        start_assistant_response = "<think>"
        if rag_info:
            temp = f"<reference_info>{rag_info}</reference_info>"
            start_assistant_response = temp + start_assistant_response

        if image:
            image_base64 = pil_to_base64(image)
            if not image_base64:
                raise ValueError("Failed to process the provided image")
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": content},
                    ]
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        
        # Extract configuration from beam search parameters
        max_chunks = beam_search_params.get("max_chunks", CONFIG["default_params"]["max_chunks"])
        num_beams = beam_search_params.get("num_beams", CONFIG["default_params"]["num_beams"])
        num_candidates = beam_search_params.get("num_candidates", CONFIG["default_params"]["num_candidates"])
        chunk_L = beam_search_params.get("chunk_L", CONFIG["default_params"]["chunk_L"])
        enable_separator = beam_search_params.get("enable_separator", "")
        
        logger.info(f"Using beam search params: max_chunks={max_chunks}, num_beams={num_beams}, "
                   f"num_candidates={num_candidates}, chunk_L={chunk_L}")
        
        # Create a copy of sampling parameters
        temp_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=chunk_L,
            n=num_candidates,
            include_stop_str_in_output=True,
        )
        
        max_new_tokens = sampling_params.max_tokens
        
        if enable_separator:
            temp_sampling_params.stop = [enable_separator]
        
        final_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=max_new_tokens,
            n=1,
            include_stop_str_in_output=True,
        )


        # Initialize search state
        previous_texts = [
            {
                "messages": messages,
                "generated_str": start_assistant_response, # Start with think token
                "generated_response": start_assistant_response,
                "score": {"score": -100},
            },
        ] if messages else []
        
        finished_texts = []
                
        # Start search process
        step = 0
        while step < max_chunks and previous_texts:
            logger.info(f"Starting step {step}")
            
            # Select top num_beams from candidate queue for next round
            llm_inputs = []
            for i in range(num_beams):
                if i < len(previous_texts):
                    llm_inputs.append(previous_texts[i])
                else :
                    llm_inputs.append(previous_texts[0])
            
            if not llm_inputs:
                logger.warning("No inputs available for generation")
                break
            
            # Generate next token segment with async model
            logger.info(f"Generating outputs for step {step}")
            try:
                outputs = await model.generate(
                    llm_inputs,
                    sampling_params=temp_sampling_params,
                    inference_name=inference_name,
                )
            except Exception as e:
                logger.error(f"Error in generation step {step}: {str(e)}")
                break
            
            candidates = []
            
            # Process outputs
            for i, output in enumerate(outputs):
                if i >= len(llm_inputs):
                    logger.warning(f"Unexpected output index: {i}, inputs: {len(llm_inputs)}")
                    continue
                    
                for j in range(len(output.choices)):
                    try:
                        output_text = output.choices[j].message.content 
                        generated_response = llm_inputs[i]["generated_response"] + output_text
                        
                        # Check if finished
                        finish = False
                        finish_reason = output.choices[j].finish_reason
                        stop_reason = getattr(output.choices[j], 'stop_reason', None)
                        
                        if output_text == "" and not(finish_reason == "stop" and stop_reason):
                            finish = True

                        if finish_reason == "stop":
                            if stop_reason == "</think>":
                                finish = True
                            if stop_reason:
                                output_text += stop_reason 
                                generated_response += stop_reason 
                                
                        # Check token limit
                        tokens = len(model.tokenizer.encode(generated_response))
                        if tokens > max_new_tokens:
                            logger.info(f"Candidate exceeded token limit: {tokens} > {max_new_tokens}")
                            finish = True
                        
                        candidates.append({
                            "messages": llm_inputs[i]["messages"],
                            "generated_str": output_text if step != 0 else "<think>" + output_text, # Add think token at the step = 0 because the first <think> token is not generated by the model
                            "generated_response": generated_response,
                            "finished": finish,
                        })
                    except Exception as e:
                        logger.error(f"Error processing candidate {j} in step {step}: {str(e)}")
            
            if not candidates:
                logger.warning(f"No candidates generated in step {step}")
                break
            
            # Extract generated responses for scoring
            generated_responses = [d["generated_response"] for d in candidates]
            
            # Calculate value model scores asynchronously
            logger.info(f"Calculating value scores for {len(generated_responses)} candidates")
            try:
                scores = await value_model.get_reward(messages, generated_responses, setting)
            except Exception as e:
                logger.error(f"Error in value scoring: {str(e)}")
                break
            
            # Add scores to candidates
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    candidate_copy = candidate.copy()
                    candidate_copy["score"] = scores[i]
                    scored_candidates.append(candidate_copy)
            
            # Process finished generations
            newly_finished = 0
            for candidate in scored_candidates:
                if candidate["finished"]:
                    finished_texts.append(candidate)
                    newly_finished += 1
            
            unfinished_candidates = [d for d in scored_candidates if not d["finished"]]
            logger.info(f"{newly_finished} candidates finished, {len(unfinished_candidates)} remaining")
            
            # Sort by score and keep top num_beams
            if unfinished_candidates:
                unfinished_candidates.sort(key=lambda x: x["score"]["score"], reverse=True)
                previous_texts = unfinished_candidates[:num_beams]
            else:
                previous_texts = []
            
            previous_texts_with_finished_candidates = [d for d in scored_candidates]
            previous_texts_with_finished_candidates.sort(key=lambda x: x["score"]["score"], reverse=True)
            previous_texts_with_finished_candidates = previous_texts_with_finished_candidates[:num_beams]
            
            # Store current step candidates (serializable version)
            step_candidates = []
            selected_num = 0
            is_finish_step = (previous_texts == [] or step + 1 >= max_chunks)
            for i, candidate in enumerate(scored_candidates):
                try:
                    step_candidates.append(
                        {
                            "id": i,
                            # "content": candidate["generated_response"],
                            "generated_str": candidate["generated_str"],
                            "value_model": {
                                "safety": candidate["score"]["safety"],
                                "value": candidate["score"]["value"],
                                "knowledge": candidate["score"]["knowledge"],
                            },
                            "score": candidate["score"]["score"],
                            "finish": candidate["finished"],
                            "select": ((scored_candidates[i] in previous_texts and not is_finish_step) or (is_finish_step and scored_candidates[i] in previous_texts_with_finished_candidates))
                                    and selected_num < num_beams,
                                    # 被选中的逻辑如下：
                                    # 若当前步骤是最后一步，则从所有完成和未完成的候选中进行排序
                                    # 否则，只从未完成的候选中进行排序
                        }
                    )
                    if step_candidates[-1]["select"]:
                        selected_num += 1
                        logger.info(f"Selected candidate {i} with score {candidate['score']['score']}")
                        logger.info(f"Generated response: {candidate['generated_str']}")
                except (KeyError, TypeError) as e:
                    logger.error(f"Error serializing candidate {i}: {str(e)}")
                        
            # Find the best candidate at this step for streaming
            try:
                # Stream the current best result
                # Calculate the number of selected candidates
                num_selected = sum(1 for d in step_candidates if d["select"])

                current_result = {
                    "step": step,
                    "finish": is_finish_step,
                    "answer": "",
                    "thinking": "",
                    "media_info": media_info,
                    "reasoning_chunks": step_candidates,
                    "inference_model_name": inference_name,
                    "source_urls": rag_result["urls"],
                    "rag_info": rag_result["info"],
                    "setting": setting,
                }
                
                yield "data: " + json.dumps(current_result, ensure_ascii=False) + "\n\n"

            except Exception as e:
                logger.error(f"Error streaming step result: {str(e)}")

            
            step += 1
        
        # Process final results
        if previous_texts:  # Unfinished texts, end reason is maximum steps reached
            logger.info(f"Adding {len(previous_texts)} unfinished texts to finished_texts")
            finished_texts.extend(previous_texts)
        
        if finished_texts:
            try:
                finished_texts.sort(key=lambda x: x["score"]["score"], reverse=True)
                if "</think>" not in finished_texts[0]['generated_response']:
                    finished_texts[0]['generated_response'] += "</think>"
                
                if inference_name == "InternVL2.5-78B":
                    finished_texts[0]['generated_response'] = finished_texts[0]['generated_response'] + "<answer>"
                
                logger.info(f"Best response has score {finished_texts[0]['score']}")
                logger.info(f"Best response: {finished_texts[0]['generated_response']}")
            except (IndexError, KeyError) as e:
                logger.error(f"Error selecting best response: {str(e)}")
        else:
            logger.warning("No finished texts available")
            finished_texts = [
                {
                    "messages": messages,
                    "generated_response": "<think></think>",
                    "score": {"score": -100},
                }
            ]

        try:
            # Generate final answer using the best thinking path
            if finished_texts:
                outputs = await model.generate(
                    finished_texts[0:1],
                    sampling_params=final_sampling_params,
                    final_think=True,
                    inference_name=inference_name,
                )
                final_response = outputs[0].choices[0].message.content.strip()
                if final_response.startswith("<answer>"):
                    final_response = final_response[len("<answer>"):]
                if final_response.endswith("</answer>"):
                    final_response = final_response[:-len("</answer>")]

                final_response = final_response.split("You are an AI assistant")[0]
                if final_response.strip() == "":
                    final_response = "N/A"
                logger.info(f"Final response generated: {final_response}")
            else:
                final_response = "Error: No finished texts available"
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            final_response = "An error occurred while generating the final response."
        
        source_urls = rag_result["urls"]
        rag_info = rag_result["info"]
        # Send final result
        final_result = {
            "step": step,
            "finish": True,
            "answer": final_response,
            "media_info": media_info,
            "thinking": finished_texts[0]["generated_response"],
            "reasoning_chunks": [],
            "inference_model_name": inference_name,
            "source_urls": source_urls,  # 添加来源URL
            "rag_info": rag_info,
            "setting": setting,
        }
        yield "data: " + json.dumps(final_result, ensure_ascii=False) + "\n\n"
    
    except Exception as e:
        logger.error(f"Error in value guidance search: {str(e)}", exc_info=True)
        error_result = {
            "error": str(e),
        }
        yield "data: " + json.dumps(error_result, ensure_ascii=False) + "\n\n"

async def validate_request_data(data: GenerateRequest):
    """Validate and extract request parameters with proper defaults"""
    if not data:
        raise ValueError("Request body cannot be empty")

    # Extract required fields with fallbacks
    question = data.question or data.message or data.prompt
    if not question:
        raise ValueError("Question/message/prompt is required")

    image_url = None
    media_info = None
    w = None
    h = None
    etag=None
    typei=None

    mediaprompt_raw = data.mediaprompt
    try:
        mediaprompt_list = json.loads(mediaprompt_raw)
        if isinstance(mediaprompt_list, list) and len(mediaprompt_list) > 0:
            image_url = mediaprompt_list[0].get("url")
            w = mediaprompt_list[0].get("w")
            h = mediaprompt_list[0].get("h")
            etag = mediaprompt_list[0].get("etag")
            typei = mediaprompt_list[0].get("type")

            media_info = {
                "req": []
            }

            media_info["req"].append({
                "url": image_url,
                "w": w,
                "h": h,
                "etag": etag,
                "type": typei,
            })
    except Exception as e:
        # 你可以选择打印或记录日志
        # load from data.get("image", "")
        logger.error(f"Error fetching image from 'mediaprompt': {str(e)}")

    if image_url is None:
        image_url = data.image or data.image_url

    # Extract optional parameters with defaults
    defaults = CONFIG["default_params"]

    # Sampling parameters
    temperature = float(data.temperature)
    top_p = float(data.top_p)
    top_k = int(data.top_k)
    max_tokens = int(data.max_tokens)

    # Beam search parameters
    chunk_L = int(data.chunk_L)
    num_beams = int(data.num_beams)
    num_candidates = int(data.num_candidates)
    max_chunks = int(data.max_chunks)

    # Construct parameter dictionaries
    beam_search_params = {
        "max_chunks": max_chunks,
        "num_beams": num_beams,
        "num_candidates": num_candidates,
        "chunk_L": chunk_L
    }

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )

    return question, image_url, beam_search_params, media_info, sampling_params


@app.post('/generateAll')
async def generate_all(request: GenerateRequest):
    """API endpoint to generate streaming responses with value guidance"""
    try:
        data = request
        prompt, image_url, beam_search_params, media_info, sampling_params = await validate_request_data(data)
        inference_name = data.inference_model_name
        setting = data.setting
        enable_rag = data.enable_rag
        logger.info(f"Received streaming request with question: {prompt} from model: {inference_name} with setting: {setting}")

        image = None
        logger.info("Fetching image...")
        try:
            logger.info(f"Fetching image from {image_url[:100]}")
            image = fetch_image({"image": image_url})
        except Exception as e:
            logger.error(f"Error fetching image: {str(e)}")
            logger.error("We will ignore the image and continue processing the text.")
        logger.info("Image fetched successfully.")

        async def generate_stream():
            accumulated_data = []

            try:
                async for item in value_guidance_search_inference_stream(
                    question=prompt,
                    image=image,
                    media_info=media_info,
                    beam_search_params=beam_search_params,
                    sampling_params=sampling_params,
                    inference_name=inference_name,
                    setting=setting,
                    enable_rag=enable_rag,
                ):
                    if isinstance(item, str) and item.strip().startswith("data:"):
                        try:
                            json_str = item.strip().removeprefix("data:").strip()
                            data_obj = json.loads(json_str)
                            # 集成 media_info
                            data_obj["media_info"] = media_info.copy() if media_info else {}
                            accumulated_data.append(data_obj)

                            sse_payload = {
                                "code": 1,
                                "msg": "",
                                "data": {
                                    "result": accumulated_data,
                                    "media_info": media_info.copy() if media_info else {},
                                }
                            }
                            yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n"
                            await asyncio.sleep(2)  # 使用异步sleep
                        except Exception as parse_error:
                            logger.warning(f"跳过格式不正确的数据项: {parse_error}")
                    else:
                        logger.warning(f"未知的数据格式: {item}")

                yield f"data: {json.dumps({'code': 0, 'msg': 'success', 'data': {{'result': accumulated_data, 'media_info': media_info.copy() if media_info else {{}}}}}, ensure_ascii=False)}\n\n"
            


            except Exception as e:
                logger.error(f"流式处理错误: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'code': -1, 'msg': str(e), 'data': []}, ensure_ascii=False)}\n\n"
            finally:
                if image and hasattr(image, 'close'):
                    image.close()
                    
        # 返回流式响应
        return StreamingResponse(
            generate_stream(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post('/generate')
async def generate(request: GenerateRequest):
    """API endpoint to generate streaming responses with value guidance"""
    try:
        # Parse and validate request data
        data = request
        question, image_url, beam_search_params, media_info, sampling_params = await validate_request_data(data)
        
        inference_name = data.inference_model_name
        setting = data.setting
        enable_rag = data.enable_rag
        logger.info(f"Received streaming request with question: {question} from model: {inference_name} with setting: {setting}")
        
        # Fetch image with error handling
        image = None
        try:
            image = fetch_image({"image": image_url})
        except Exception as e:
            logger.error(f"Error fetching image: {str(e)}")
            logger.error(f"We will ignore the image and continue processing the text.")        
        
        async def generate_stream():
            try:
                async for item in value_guidance_search_inference_stream(
                    question=question,
                    image=image,
                    media_info=media_info,
                    beam_search_params=beam_search_params,
                    sampling_params=sampling_params,
                    inference_name=inference_name,
                    setting=setting,
                    enable_rag=enable_rag,
                ):
                    yield item
            finally:
                # 清理资源
                if image and hasattr(image, 'close'):
                    image.close()
        
        # 返回流式响应
        return StreamingResponse(
            generate_stream(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    
    except ValueError as e:
        # Handle validation errors
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get('/health')
async def health_check():
    if model and value_model:
        return {
            "status": "healthy", 
            "models_loaded": True,
            "inference_endpoint": CONFIG["model_endpoints"]["inference"],
            "gating_endpoint": CONFIG["model_endpoints"]["gating"],
            "value_endpoints": {
                "safety": CONFIG["model_endpoints"]["safety"],
                "value": CONFIG["model_endpoints"]["value"],
                "knowledge": CONFIG["model_endpoints"]["knowledge"]
            }
        }
    else:
        return JSONResponse(
            content={"status": "unhealthy", "models_loaded": False},
            status_code=503
        )


@app.post('/step_generate')
async def step_generate(request: GenerateRequest):
    """API endpoint to generate a specific step response with value guidance"""
    try:
        # Parse and validate request data
        data = request
        question, image_url, beam_search_params, media_info, sampling_params = await validate_request_data(data)
        
        # Extract additional required parameters for step-wise generation
        generated_response = data.generated_response
        inference_name = data.inference_model_name
        step = data.step
        if step < 0:
            return JSONResponse(status_code=400, content={"error": "Step cannot be negative"})
        
        max_chunks = beam_search_params.get("max_chunks", CONFIG["default_params"]["max_chunks"])
        # Check if we should generate final response instead of chunks
        is_final_step = False
        final_step_reason = ""
        
        # Check if we've reached max chunks
        if step >= max_chunks:
            is_final_step = True
            final_step_reason = "max_chunks_reached"
            
        # Check if generated_response exceeds max tokens
        if len(model.tokenizer.encode(generated_response)) > sampling_params.max_tokens:
            is_final_step = True
            final_step_reason = "max_tokens_exceeded"
            
        # Check if response already contains stop token
        if "</think>" in generated_response :
            is_final_step = True
            final_step_reason = "stop_token_found"
            
        logger.info(f"Received step generation request, step: {step}, question: {question}, from {inference_name}, is_final: {is_final_step}, reason: {final_step_reason}")
        
        # Fetch image with error handling 
        image = None
        try:
            if image_url:
                image = fetch_image({"image": image_url})
        except Exception as e:
            logger.error(f"Error fetching image: {str(e)}")
            logger.error(f"We will ignore the image and continue processing the text.")
        
        # Function to handle the async step generation
        async def generate_step_with_cleanup():
            try:
                # 选择使用哪个异步生成器
                if is_final_step:
                    async_gen = generate_final_response(
                        step=step,
                        question=question,
                        image_url=image_url,
                        image=image,
                        media_info=media_info,
                        generated_response=generated_response,
                        final_step_reason=final_step_reason,
                        sampling_params=sampling_params,
                        inference_name=inference_name,
                    )
                else:
                    async_gen = step_value_guidance_inference(
                        question=question,
                        image_url=image_url,
                        image=image,
                        media_info=media_info,
                        generated_response=generated_response,
                        step=step,
                        beam_search_params=beam_search_params,
                        sampling_params=sampling_params,
                        inference_name=inference_name,
                    )
                
                # 直接从异步生成器中产出结果
                async for item in async_gen:
                    yield item
            
            except Exception as e:
                logger.error(f"Error in step streaming: {str(e)}", exc_info=True)
                yield json.dumps({
                    "error": str(e),
                    "content": "An error occurred during step processing."
                }) + "\n"
            finally:
                # 清理图像资源
                if image:
                    try:
                        if hasattr(image, 'close'):
                            image.close()
                    except:
                        pass
            
        return StreamingResponse(
            generate_step_with_cleanup(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    
    except ValueError as e:
        # Handle validation errors
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error in step_generate endpoint: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


async def generate_final_response(step, question, image_url, image, media_info, generated_response, final_step_reason, sampling_params, inference_name="sft"):
    """Generate the final response when termination conditions are met"""
    logger.info(f"Generating final response, reason: {final_step_reason}")
    logger.info(f"Get ImageURL:{image_url[:100]}...\nGenerated_Response:{generated_response}" )
    logger.info(f"SamplingParams:{sampling_params}")

    try:
        # Prepare input
        content = PROMPT_TEMPLATE.format(question=question)
        if image:
            image_base64 = pil_to_base64(image)
            if not image_base64:
                raise ValueError("Failed to process the provided image")
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": content},
                    ]
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        
        # Ensure the generated response has the closing think tag
        if "</think>" not in generated_response:
            generated_response += "</think>"
        
        if "<think>" not in generated_response:
            generated_response = "<think>" + generated_response
            
        # Create input state for final response generation
        final_input = {
            "messages": messages,
            "generated_response": generated_response,
            "score": {"score": 0}  # Dummy score for final response
        }
        
        # Create sampling params for final response
        final_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=sampling_params.max_tokens,
            n=1,
            include_stop_str_in_output=True,
        )
        
        # Generate final answer
        try:
            outputs = await model.generate(
                [final_input],
                sampling_params=final_sampling_params,
                final_think=True,
                inference_name=inference_name,
            )
            final_response = outputs[0].choices[0].message.content
            # Clean up the response by removing any AI assistant preamble
            final_response = final_response.split("You are an AI assistant")[0].strip()
            logger.info(f"Final response generated: {final_response}")
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            final_response = "An error occurred while generating the final response."
        
        # Send the final result
        final_result = {
            "step": step,
            "finish": True,
            "question": question,
            "image_url": image_url,
            "answer": final_response,
            "thinking": generated_response,
            "reasoning_chunks": [],
            "media_info": media_info,
            # "inference_model_name": inference_name,
            "source_urls": [],
            "rag_info": "",
        }
        extra_info = {
             "code": 0,
             "msg": "success",
             "data": final_result
        }

        await asyncio.sleep(2)
        yield f"data: {json.dumps(extra_info, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Error generating final response: {str(e)}", exc_info=True)
        error_result = {
            "error": str(e),
            "is_final": True
        }
        yield "data: " + json.dumps(error_result, ensure_ascii=False) + "\n\n"


async def step_value_guidance_inference(question, image_url, image, media_info, generated_response, step, beam_search_params, sampling_params, inference_name="sft"):
    """Generator function for streaming a specific step in value guidance search"""
    logger.info(f"Starting step {step} generation for question: {question}")
    logger.info(f"Get ImageURL:{image_url[:100]}..." )
    logger.info(f"Generated_Response:{generated_response}")
    
    try:
        # Prepare input
        rag_info = ""
        source_urls = []

        content = PROMPT_TEMPLATE.format(question=question)

        if image:
            image_base64 = pil_to_base64(image)
            if not image_base64:
                raise ValueError("Failed to process the provided image")
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": content},
                    ]
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        
        # Extract configuration from beam search parameters
        num_beams = beam_search_params.get("num_beams", CONFIG["default_params"]["num_beams"])
        num_candidates = beam_search_params.get("num_candidates", CONFIG["default_params"]["num_candidates"])
        chunk_L = beam_search_params.get("chunk_L", CONFIG["default_params"]["chunk_L"])
        enable_separator = beam_search_params.get("enable_separator", "")
        
        logger.info(f"Using beam search params for step {step}: num_beams={num_beams}, "
                   f"num_candidates={num_candidates}, chunk_L={chunk_L}")
        
        # Create a copy of sampling parameters for chunk generation
        temp_sampling_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=chunk_L,
            n=num_candidates,
            include_stop_str_in_output=True,
        )
        
        if enable_separator:
            temp_sampling_params.stop = [enable_separator]
        
        # Initialize with the current state based on generated_response
        if not generated_response.startswith("<think>"):
            generated_response = "<think>" + generated_response
        initial_state = {
            "messages": messages,
            "generated_str": "",  # Will be filled by the current step generation
            "generated_response": generated_response,
            "score": {"score": -100},
        }
        
        # Generate next token segment for this specific step
        logger.info(f"Generating outputs for step {step}")
        try:
            outputs = await model.generate(
                [initial_state],
                sampling_params=temp_sampling_params,
                inference_name=inference_name,
            )
        except Exception as e:
            logger.error(f"Error in generation step {step}: {str(e)}")
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
            return
        
        candidates = []
        
        # Process outputs from the model
        for i, output in enumerate(outputs):
            for j in range(len(output.choices)):
                try:
                    output_text = output.choices[j].message.content 
                    new_generated_response = generated_response + output_text
                    if step == 0:
                        output_text = "<think>" + output_text

                    # Check if this chunk completes the generation
                    finish = False
                    finish_reason = output.choices[j].finish_reason
                    stop_reason = getattr(output.choices[j], 'stop_reason', None)
                    
                    if output_text == "":
                        finish = True

                    if finish_reason == "stop":
                        if stop_reason == "</think>":
                            finish = True
                        if stop_reason:
                            output_text += stop_reason 
                            new_generated_response += stop_reason 
                            
                    # Check token limit
                    tokens = len(model.tokenizer.encode(new_generated_response))
                    if tokens > sampling_params.max_tokens:
                        logger.info(f"Candidate exceeded token limit: {tokens} > {sampling_params.max_tokens}")
                        finish = True
                    
                    candidates.append({
                        "messages": messages,
                        "generated_str": output_text,
                        "generated_response": new_generated_response,
                        "finished": finish,
                    })
                except Exception as e:
                    logger.error(f"Error processing candidate {j} in step {step}: {str(e)}")
        
        if not candidates:
            logger.warning(f"No candidates generated in step {step}")
            yield "data: " + json.dumps({"error": "No candidates generated"}) + "\n\n"
            return
        
        # Extract generated responses for scoring
        generated_responses = [d["generated_response"] for d in candidates]
        
        # Calculate value model scores asynchronously
        logger.info(f"Calculating value scores for {len(generated_responses)} candidates")
        try:
            scores = await value_model.get_reward(messages, generated_responses)
        except Exception as e:
            logger.error(f"Error in value scoring: {str(e)}")
            yield "data: " + json.dumps({"error": f"Error in value scoring: {str(e)}"}) + "\n\n"
            return
        
        # Add scores to candidates
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            if i < len(scores):
                candidate_copy = candidate.copy()
                candidate_copy["score"] = scores[i]
                scored_candidates.append(candidate_copy)
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x["score"]["score"], reverse=True)
        
        # Prepare the output format
        step_candidates = []
        for i, candidate in enumerate(scored_candidates):
            try:
                step_candidates.append({
                    "id": i,
                    "content": candidate["generated_str"],
                    "generated_response": candidate["generated_response"],
                    "value_mode": {
                        "safety": candidate["score"]["safety"],
                        "value": candidate["score"]["value"],
                        "knowledge": candidate["score"]["knowledge"],
                    },
                    "score": candidate["score"]["score"],
                    "finish": candidate["finished"],
                })
            except (KeyError, TypeError) as e:
                logger.error(f"Error serializing candidate {i}: {str(e)}")
        
        # Stream the result
        result = {
            "step": step,
            "question": question,
            "image_url": image_url,
            "finish": False,
            "answer": "",
            "reasoning_chunks": step_candidates,
            "media_info": media_info,
            # "inference_model_name": inference_name,
            "source_urls": source_urls,
            "rag_info": rag_info if step == 0 else ""
        }
        
        extra_info = {
             "code": 0,
             "msg": "success",
             "data": result
        }
        # data CODE 0 MSG SUCEESS DATA:
        await asyncio.sleep(2)
        yield f"data: {json.dumps(extra_info, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in step value guidance: {str(e)}", exc_info=True)
        error_result = {
            "error": str(e),
        }
        yield "data: " + json.dumps(error_result, ensure_ascii=False) + "\n\n"



@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    logger.info("Loading models on startup...")
    load_models()
    
    logger.info("Model endpoints configuration:")
    logger.info(f"Inference model: {CONFIG['model_endpoints']['inference']}")
    logger.info(f"Gating endpoint: {CONFIG['model_endpoints']['gating']}")
    logger.info(f"Value models: {CONFIG['model_endpoints']['safety']}, "
               f"{CONFIG['model_endpoints']['value']}, "
               f"{CONFIG['model_endpoints']['knowledge']}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the async value guidance API server')
    parser.add_argument('--inference_model_url', type=str, help='URL for inference model API', default="")
    parser.add_argument('--safety_model_url', type=str, help='URL for safety model API', default="")
    parser.add_argument('--value_model_url', type=str, help='URL for value model API', default="")
    parser.add_argument('--knowledge_model_url', type=str, help='URL for knowledge model API', default="")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=9002, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads')
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.inference_model_url:
        CONFIG["model_endpoints"]["inference"] = args.inference_model_url
    if args.safety_model_url:
        CONFIG["model_endpoints"]["safety"] = args.safety_model_url
    if args.value_model_url:
        CONFIG["model_endpoints"]["value"] = args.value_model_url
    if args.knowledge_model_url:
        CONFIG["model_endpoints"]["knowledge"] = args.knowledge_model_url
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)

"""    
curl -X POST "http://localhost:9002/generate" -H "Content-Type: application/json" -d '{
    "question": "What is shown in this image? think in <think></think>, and put your answer in the <answer></answer>",
    "image_url": "https://docs.vllm.ai/en/stable/_static/vllm-logo-text-light.png",
    "max_chunks": 5,
    "num_beams": 1,
    "num_candidates": 3,
    "chunk_L": 100,
    "temperature": 0.7,
    "top_k": 50,
    "max_tokens": 2048,
    "enable_rag": false,
    "enable_gating": false
}'
"""
