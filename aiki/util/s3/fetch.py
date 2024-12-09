# -*- coding: utf-8 -*-
import gzip
import io
import json
import shutil
from petrel_client.client import Client

client = Client('~/petreloss.conf')
project_name = "ey"
url = "s3://llm-pipeline/zh-pdf-ey/epdfb@001_epdfc@001_clean@002_2024m6_dedup@0.7/"
path = f"/mnt/hwfile/kilab/leishanzhe/data/{project_name}/"
data = client.get(url)

# # 将数据写入本地文件
# with open('part-6656fc2cb915-005243.jsonl.gz', 'wb') as f:
#     f.write(data)

print(client.isdir(url))
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