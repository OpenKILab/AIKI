from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch
from bson import ObjectId
from openai import OpenAI
import streamlit as st
from streamlit_timeline import st_timeline

import base64

from transformers import AutoModel

from aiki.database import JSONFileDB
from aiki.database.chroma import ChromaDB
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal import MultiModalProcessor, ModalityType, TextHandler, VectorHandler, TextModalityData
from aiki.retriever.retriever import DenseRetriever

model = SentenceTransformer('lier007/xiaobu-embedding-v2')

def get_data_uri(filepath):
    binary_file_content = open(filepath, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_file_content).decode('utf-8')

    ext = filepath.split('.')[-1]
    data_uri = f'data:image/{ext};base64,{base64_utf8_str}'
    return data_uri

st.title("Demo")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input := st.chat_input():
    client = OpenAI(
        base_url="https://api.claudeshop.top/v1")
    st.chat_message("user").write(input)

    processor = MultiModalProcessor()
    source_db = JSONFileDB("./db/flicker8k_xiaobu/flicker8k.json")
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/flicker8k_xiaobu/flicker8k_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db,
                                                                  embedding_func=model.encode))

    dense_retriever = DenseRetriever(processor=processor)
    print(input )
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            # content="我儿子玩雪玩的很开心",
            content=input,
            _id=ObjectId(),
            metadata={}
        )
    ])
    result = dense_retriever.search(retrieval_data, num=3)
    # 根据检索到的结果回复
    prompt = (f"Here are some images related to the query:"
              f"\n1. 时间: {result.items[0].metadata['timestamp']} 内容：{result.items[0].metadata['summary']}"
              f"\n2. 时间: {result.items[1].metadata['timestamp']} 内容：{result.items[1].metadata['summary']}"
              f"\n3. 时间: {result.items[2].metadata['timestamp']} 内容：{result.items[2].metadata['summary']}"
              f"\n根据这些图片来回复用户的问题：{input}。")
    print(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    # 输出timeline
    items = []
    for item in result.items:
        # print(item.metadata)
        items.append(
            {"id": str(item._id),
             "content": f'<div>{item.metadata["summary"][:16]}...</div><img src="data:image/jpg;base64,{item.content}" style="width:128px; height:100px;">',
             "start": datetime.fromtimestamp(item.metadata['timestamp']).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
             }
        )
    timeline = st_timeline(items, groups=[], options={}, height="512px")
    # st.write(timeline)