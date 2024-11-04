import base64
from openai import OpenAI
import openai
import os

client = OpenAI(
    base_url="https://api.claudeshop.top/v1"
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
base_path = os.getcwd()
path = os.path.join(base_path, "aiki", "resource", "source", "imgs", "外滩路人.jpg")
image_path = path
print(path)
# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "What is in this image?",
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
            },
            },
        ],
        }
    ],
)

print(response.choices[0].message.content)
