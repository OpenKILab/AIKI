from datetime import datetime, timedelta
import os
import random
import shutil

from bson import ObjectId
from sentence_transformers import SentenceTransformer

from aiki.database.chroma import ChromaDB
from aiki.database.json_file import JSONFileDB
from aiki.indexer.indexer import MultimodalIndexer, encode_image_to_base64
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType
from aiki.multimodal.base import ModalityType, MultiModalProcessor
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextHandler
from aiki.multimodal.vector import VectorHandler

# Define paths
dataset_path = "/Users/mac/Documents/pjlab/repo/flickr8k/Flicker8k_Dataset"
caption_file = "resource/flicker8k/caption.txt"
validation_folder = "resource/flicker8k/validation"

# Create validation folder if it doesn't exist
os.makedirs(validation_folder, exist_ok=True)

# Read the caption file and extract filenames
with open(caption_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Extract filenames from the caption file
filenames = []
for line in lines:
    filenames_and_descriptions = []
    for line in lines:
        if line.strip() and not line.startswith('|'):
            parts = line.split('\t')
            if len(parts) > 1:
                filename = parts[0].strip()
                description = parts[1].strip()  # Extract the description part
                filenames_and_descriptions.append((filename, description))


for filename, description in filenames_and_descriptions:
    print(f"Filename: {filename}, Description: {description}")
    model = SentenceTransformer('lier007/xiaobu-embedding-v2')

    source_db = JSONFileDB("./db/flicker8k_xiaobu/flicker8k.json")
    
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/flicker8k_xiaobu/flicker8k_index")

    processor = MultiModalProcessor()

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db, embedding_func=model.encode))
    
    multimodal_indexer = MultimodalIndexer(processor=processor)
    
    base_path = os.getcwd()

    print("Current file name:", filename)
    file_path = f"{validation_folder}/{filename}.jpg"
    encoded_image = encode_image_to_base64(file_path)
    
    retrieval_data = RetrievalData(
        items=[
            ImageModalityData(
                content= encoded_image,
                _id = ObjectId(),
                metadata={
                    "timestamp": int(datetime(
                        2024, 11, random.randint(1, 13), 
                        random.randint(6, 20), 
                        random.randint(0, 59), 
                        random.randint(0, 59)
                    ).timestamp()),
                    "summary": description
                }
        ),
        ]
    )

    # Index the data
    multimodal_indexer.index(retrieval_data)
