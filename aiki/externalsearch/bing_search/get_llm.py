import requests
from openai import OpenAI
from httpx import Client

base_url = "http://0.0.0.0:10022/v1"  
api_key = "sk-123456" 
model_name = "gemma-3-27b-it"

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=Client(verify=False), 
)


def generate_text(prompt, max_tokens=3042, temperature=0.7):
    """
    使用 vLLM 服务生成文本
    
    Args:
        prompt (str): 输入提示
        max_tokens (int): 最大生成令牌数
        temperature (float): 生成多样性参数
        
    Returns:
        str: 生成的文本
    """
    try:
        messages = [
            
            ]
        messages.extend(prompt)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

def generate_query(prompt, max_tokens=6042, temperature=0.7):
    """
    使用 vLLM 服务生成文本
    
    Args:
        prompt (str): 输入提示
        max_tokens (int): 最大生成令牌数
        temperature (float): 生成多样性参数
        
    Returns:
        str: 生成的文本
    """
    try:
        # messages = [
            
        #     ]
        # messages.extend(prompt)
        messages = [
            {"role": "system", "content": "假设你是一个多学科专家，你有一次运用互联网搜索工具的机会，你现在需要根据以下问题构造一个互联网搜索关键句，以便搜索到能解决该问题的知识点。将搜索关键句以/boxed{}格式返回"},
            {"role": "user", "content": prompt},
            ]
        # print(messages)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return None
    
def generate_summary(prompt, max_tokens=10042, temperature=0.7):
    try:
        messages = [
            {"role": "system", "content": "假设你是一个多学科总结专家，你不需要回答问题，你现在需要根据\"问题\"和\"相关信息\"，构造一个总结，该总结可以帮助回答这个问题，这个总结是简洁的，专业的，完全突出了知识点。直接返总结内容。"},
            {"role": "user", "content": prompt},
            ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return None