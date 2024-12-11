# -*- coding: utf-8 -*-
import gzip
import io
import json
import os
import shutil
from PIL import Image 
from petrel_client.client import Client

client = Client('~/petreloss.conf')
# url = 's3://llm-pipeline-media/pdf-imgs/b777312a0dd842211f1ba8fbcd23711f70d763f007c229c40c240be3da4bc4ae.jpg'
# url = "s3://llm-pipeline-media/pdf-imgs/cbd26093234ea3aa2cc7a80fe2a99e59c86261de6e9f8e09873682b1b66f0e5e.jpg"
# data = client.get(url)
# print(url)
# print(type(data))

# # 将二进制数据转换为字节流
# image_stream = io.BytesIO(data)

# # 使用 Pillow 打开图像
# try:
#     with Image.open(image_stream) as img:
#         # 获取图像格式
#         image_format = img.format
#         print(f'The image format is: {image_format}')
# except IOError:
#     print("The data provided is not a valid image.")


client = Client('~/petreloss.conf')
project_name = "stcn"
url = "s3://llm-pipeline/zh-zhengquan-stcn/clean@002_2024m6_dedup@0.7/"
path = f"/mnt/hwfile/kilab/leishanzhe/data/{project_name}/"
if not os.path.exists(path):
    os.makedirs(path)
data = client.get(url)

# # 将数据写入本地文件
# with open('part-6656fc2cb915-005243.jsonl.gz', 'wb') as f:
#     f.write(data)

print(client.isdir(url))
print(data)
contents = client.list(url)
for content in contents:
    if content.endswith('/'):
        print('directory:', content)
    else:
        object_file_path = f'{path}{content}'
        data = client.get(f"{url}{content}")
        with open(object_file_path, 'wb') as f:
            f.write(data)
        
        output_file = object_file_path.replace('.gz', '')
        
        with gzip.open(object_file_path, "rb") as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(content)
# print(data)