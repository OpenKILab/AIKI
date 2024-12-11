import os
import subprocess
import pandas as pd
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import ClipIndexer
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler, TextModalityData
from aiki.multimodal.vector import VectorHandler
import ijson
from tqdm import tqdm
from bson import ObjectId
from aiki.database.sqlite import SQLiteDB
from aiki.aiki import AIKI


def proxy_on():
    proxy_url = "http://wangxuhong:1ivJjHnljSrTLIRrdXTjC8slziaV3g0EMNLWqDzyfJEUDpXd2onxYt29CcUp@10.1.20.50:23128"
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url

def proxy_off():
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)

proxy_on()
ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/ey/")
proxy_off()

batch_size = 64
batch_data = []

path = '/mnt/hwfile/kilab/leishanzhe/data/ey/part-6656fc2cb915-005243.jsonl'
target_path = 'path to objects365 images folder'


with open(path, 'r', encoding='utf-8') as file:
    # Use ijson to parse the file incrementally
    for data in tqdm(ijson.items(file, '', multiple_values=True), desc="Processing data"):
        if 'content_list' in data:
            for content in data['content_list']:
                if content['type'] == 'text':
                    retrieve_data = TextModalityData(
                        _id=ObjectId(),
                        content=content['text'],
                    )
                    batch_data.append(retrieve_data)

                elif content['type'] == 'image':
                    if 'img_path' in content:
                        retrieve_data = ImageModalityData(
                            _id=ObjectId(),
                            url=content['img_path'],
                        )
                        batch_data.append(retrieve_data)

                if len(batch_data) >= batch_size:
                    retrieval_data = RetrievalData(items=batch_data)
                    ak.batch_index(retrieval_data)
                    batch_data = []
if batch_data:
    retrieval_data = RetrievalData(items=batch_data)
    ak.batch_index(retrieval_data)


# with open(path, 'r') as f:
#     objects = ijson.items(f, 'annotations.item')
    
#     for obj in tqdm(objects, desc="Processing objects"):
#         with open(path, 'r', encoding='utf-8') as file:
#             # Use ijson to parse the file incrementally
#             for data in ijson.items(file, '', multiple_values=True):
#                 if 'content_list' in data:
#                     for content in data['content_list']:
#                         if content['type'] == 'text':
#                             retrieve_data = TextModalityData(
#                                 _id = ObjectId(),
#                                 content = content['text'],
#                             )
#                             batch_data.append(retrieve_data)

#                         elif content['image'] == 'image':
#                             ...
                        
#                         if len(batch_data) >= batch_size:
#                             retrieval_data = RetrievalData(items=batch_data)
#                             multimodal_indexer.batch_index(retrieval_data)
#                             batch_data = []
#     multimodal_indexer.batch_index(retrieval_data)
