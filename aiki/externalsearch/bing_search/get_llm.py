import requests
from openai import OpenAI  # 添加 OpenAI 库的导入
from httpx import Client  # 添加 httpx 客户端的导入

# 设置服务器的 URL
base_url = "http://0.0.0.0:10022/v1"  # 更新为 OpenAI 的 URL
api_key = "sk-123456"  # 添加 API 密钥
# base_url = "https://boyuerichdata.chatgptten.com/v1"
# api_key = "sk-7js5cUGSTxEWbSW54hS2M0kqKxuFzIgkNUyDWtYvCnhltIQV"

# 创建 OpenAI 客户端实例，连接到 vLLM 服务
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=Client(verify=False),  # 如果使用自签名证书，设置为不验证
)

# 示例函数：使用 vLLM 服务生成文本
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
        print(messages)
        response = client.chat.completions.create(
            model="Qwen2.5-72B-Instruct",  # 替换为 vLLM 中部署的模型名称
            # model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

# print(generate_text("hi, how are you?"))