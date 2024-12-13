import glob
import os
import time
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
import subprocess


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
ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/wiki_full/")
proxy_off()

batch_size = 256
batch_data = []

path = '/mnt/hwfile/kilab/leishanzhe/data/wiki'
jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))

start_time = time.time()

"""
{"url":"https://zh.wikipedia.org/wiki?curid=1803908","content_list":[{"type":"text","text":"## 別爾庫米奇河","modify_it_list":["remove_tail_bracket"],"drop_it_list":[]},{"type":"text","text":"別爾庫米奇河是俄羅斯的河流，位於堪察加半島，河道全長約51公里，流域面積285平方公里，河水主要來自融雪和雨水，最終注入基列夫納河，是虹鱒和遠東紅點鮭的棲息地。","modify_it_list":[],"drop_it_list":[]}],"remark":{"cur_id":"1803908"},"data_source":"zh-baike-wiki","track_loc":["s3://llm-process-pperf/uf-zh-baike-wiki/format/v001/part-66152e490ad2-000000.jsonl?bytes=0,405","s3://llm-pipeline/zh-baike-wiki/clean@002/part-66152e490ad2-000000.jsonl.gz?bytes=0,750"],"track_id":"f89ffcaf0ca9bcb7c9efcb68bd9336daa9b5a79d","labels":{"domain_level":"","from_domestic_source":false,"unsafe_word_min_level":"L3","dup_count":0,"dup_ids":[],"dup_locs":[]},"modify_doc_list":["update_domain_level_by_data_dict","update_language_by_str","update_content_style_by_data_dict","update_format_by_data_dict","update_rules_key","remove_tail_bracket","update_from_domestic_source","update_unsafe_word_by_data_dict"],"language":"zh","content_style":"pedia","format":"text","rules_key":"pedia-zh-text","id":"BmdA7oM08EVDoYrj-gVm","doc_loc":"s3://llm-pipeline/zh-baike-wiki/clean@002_2024m6_dedup@0.7/part-66152e490ad2-000000.jsonl.gz?bytes=0,0"}
"""
for file_path in jsonl_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        # Use ijson to parse the file incrementally
        for data in tqdm(ijson.items(file, '', multiple_values=True), desc="Processing data", disable=True):
            if 'content_list' in data:
                for content in data['content_list']:
                    if content['type'] == 'text':
                        retrieve_data = TextModalityData(
                            _id=ObjectId(),
                            content=content['text'],
                        )
                        batch_data.append(retrieve_data)

                    elif content['type'] == 'image':
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

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")