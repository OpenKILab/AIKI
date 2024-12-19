# -*- coding: utf-8 -*-
import gzip
import io
import json
import os
import shutil
from PIL import Image 
from petrel_client.client import Client

client = Client('~/petreloss.conf')
project_name = "stcn"
# s3://bucket_name.ip:port/path/
url = "s3://llm-pipeline/zh-zhengquan-stcn/clean@002_2024m6_dedup@0.7/"
url = "s3://kilab-dataset/"
target_url = "s3://kilab-dataset/test.jsonl"
path = f"/mnt/hwfile/kilab/leishanzhe/data/{project_name}/"
if not os.path.exists(path):
    os.makedirs(path)
    
data = """{"id": 1, "text": "这是第一条数据", "label": "positive"}
{"id": 2, "text": "这是第二条数据", "label": "negative"}
{"id": 3, "text": "这是第三条数据", "label": "neutral"}
{"id": 4, "text": "这是第四条数据", "label": "positive"}"""

client.put(target_url, data)

data = client.get(url)

print(client.isdir(url))
contents = client.list(url)
print(contents)
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