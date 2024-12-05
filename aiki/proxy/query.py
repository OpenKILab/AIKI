from openai import OpenAI
import os
from aiki.config.config import Config

def query_llm(prompt: str) -> str:
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', '..', 'aiki', 'config', 'config.yaml')
    config = Config(config_path)
    client = OpenAI(
        base_url=config.get('base_url', "https://api.claudeshop.top/v1")
    )
    response = client.chat.completions.create(
    model=config.get('model', "gpt-4o-mini"),
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    content = response.choices[0].message.content
    return content