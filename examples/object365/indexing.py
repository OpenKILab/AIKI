import os
import pandas as pd
from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import ClipIndexer
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler
from aiki.multimodal.vector import VectorHandler
import ijson
from tqdm import tqdm
from bson import ObjectId

db_path = "/mnt/hwfile/kilab/leishanzhe/db/objects365_val/"
name = "jina_clip"
processor = MultiModalProcessor()
source_db = JSONFileDB(f"{db_path}/{name}/{name}.json")
chroma_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"{db_path}/{name}/{name}_index")

processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db))

multimodal_indexer = ClipIndexer(processor=processor)

batch_size = 16
batch_data = []

with open('/mnt/hwfile/kilab/leishanzhe/data/Objects365v1/Objects365v1/OpenDataLab___Objects365_v1/raw/Objects365_v1/2019-08-02/objects365_val.json', 'r') as f:
    objects = ijson.items(f, 'annotations.item')
    
    for obj in tqdm(objects, desc="Processing objects"):
        annot = obj
        img_id = annot['image_id']
        cat_id = annot['category_id']
        annot_id = annot['id']
        prefix = f"{img_id}_{annot_id}_{cat_id}"
        image_file_name = f"""{prefix}.jpg"""
        
        directory_path = '/mnt/hwfile/kilab/leishanzhe/data/Objects365v1/Objects365v1/OpenDataLab___Objects365_v1/raw/Objects365_v1/2019-08-02/subimages_val/'
        file_path = os.path.join(directory_path, image_file_name)
        if os.path.exists(file_path):
            image_data = ImageModalityData(
                url = file_path,
                _id = ObjectId(),
                metadata={
                    "parent": str(img_id),
                    "annot_id": str(annot_id),
                    "image_id": str(img_id),
                    "cat_id": str(cat_id),
                }
            )
            batch_data.append(image_data)
            
            # 当批量数据达到指定大小时，进行索引
            if len(batch_data) >= batch_size:
                retrieval_data = RetrievalData(items=batch_data)
                multimodal_indexer.batch_index(retrieval_data)
                batch_data = []  # 清空批量数据
        else:
            print(f"文件 {image_file_name} 不存在于目录中。")