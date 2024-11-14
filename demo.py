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

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)

def get_data_uri(filepath):
    binary_file_content = open(filepath, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_file_content).decode('utf-8')

    ext = filepath.split('.')[-1]
    data_uri = f'data:image/{ext};base64,{base64_utf8_str}'
    return data_uri

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    client = OpenAI(
        base_url="https://api.claudeshop.top/v1",
        api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)
    processor = MultiModalProcessor()
    source_db = JSONFileDB("./db/flicker8k_jina.json")
    chroma_db = ChromaDB(collection_name="text_index", persist_directory="./db/flicker8k_jina_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db,
                                                                  embedding_func=model.encode))

    dense_retriever = DenseRetriever(processor=processor)
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            # content="我儿子玩雪玩的很开心",
            content=prompt,
            _id=ObjectId(),
        )
    ])
    result = dense_retriever.search(retrieval_data, num=10)
    items = []
    for item in result.items:
        # print(item.metadata)
        items.append(
            {"id": str(item._id),
             "content": f'<div>{item.metadata["summary"][:16]}...</div><img src="data:image/jpg;base64,{item.content}" style="width:128px; height:100px;">',
             "start": item.metadata['timestamp'].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
             }
        )
    timeline = st_timeline(items, groups=[], options={}, height="512px")
    # st.write(timeline)