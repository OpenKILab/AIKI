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
    
    st.session_state.messages.append({"role": "user", "content": input})
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
    items = []
    timeline = st_timeline(items, groups=[], height="512px", options={
                        "editable": {
                "add": True,  # 是否允许通过双击添加新项目
                "updateTime": True,  # 是否允许水平拖动项目以更新时间
                "updateGroup": True,  # 是否允许将项目从一个组拖动到另一个组
                "remove": True,  # 是否允许通过点击右上角的删除按钮删除项目
                "overrideItems": True  # 是否允许这些选项覆盖项目的可编辑性
            },
            "selectable": False  # 是否允许选择项目
        })