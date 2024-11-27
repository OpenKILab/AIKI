from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch
from bson import ObjectId
from openai import OpenAI
import streamlit as st
from streamlit_timeline import st_timeline

import base64

from transformers import AutoModel

from aiki.agent.baseagent import AgentChain, Message
from aiki.database import JSONFileDB
from aiki.database.chroma import ChromaDB
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal import MultiModalProcessor, ModalityType, TextHandler, VectorHandler, TextModalityData
from aiki.retriever.retriever import DenseRetriever
from aiki.agent.baseagent import AgentChain, Message

model = SentenceTransformer('lier007/xiaobu-embedding-v2')
embedding_func = model.encode
agent_chain = AgentChain()

def get_data_uri(filepath):
    binary_file_content = open(filepath, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_file_content).decode('utf-8')

    ext = filepath.split('.')[-1]
    data_uri = f'data:image/{ext};base64,{base64_utf8_str}'
    return data_uri

st.title("Demo")

# 创建两列布局
left_column, _ = st.columns([3, 1])  # 使用占位符列

# 在左边一列放置聊天消息
with left_column:
    timeline_placeholder = st.empty()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
# 在右边侧边栏放置 proceed_bar, text_area_placeholder
with st.sidebar:
    proceed_bar = st.progress(1, text="agent progress")
    text_area_placeholder = st.empty()

    text_area_placeholder.text_area("agent status", "Initial", height=200)

    if 'talk_completed' not in st.session_state:
        st.session_state.talk_completed = False

if input := st.chat_input():
    client = OpenAI(
        base_url="https://api.claudeshop.top/v1")
    st.chat_message("user").write(input)

    name = "xiaobu_summary"
    processor = MultiModalProcessor()
    source_db = JSONFileDB(f"./db/{name}/{name}.json")
    chroma_db = ChromaDB(collection_name=f"{name}_index", persist_directory=f"./db/{name}/{name}_index")

    processor.register_handler(ModalityType.TEXT, TextHandler(database=source_db))
    processor.register_handler(ModalityType.IMAGE, TextHandler(database=source_db))
    processor.register_handler(ModalityType.VECTOR, VectorHandler(database=chroma_db,
                                                                    embedding_func=embedding_func))

    dense_retriever = DenseRetriever(processor=processor)
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
    prompt = (f"Here are some images/texts related to the query:")
    for item in result.items:
        prompt += f"\nnew images/texts : 时间: {item.metadata['timestamp']} 内容：{item.metadata.get('summary', item.content)}"
    prompt += f"\n根据这些信息，从一个旁观者的角度，根据时间内容等信息，来回复用户的问题，表述尽量类似于对话而不是对信息和问题的陈述：{input}。"
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    # 输出timeline
    items = []
    for item in result.items:
        # print(item.metadata)
        content_html = f'<div style="width:100%; height:100%; background-color:red;">' \
                        f'<div style="background-color:inherit;">{item.metadata.get("summary", item.content)[:16]}...</div>' \
                        f'</div>'
        if item.modality == ModalityType.IMAGE:  # Check if item.content is not empty
            content_html += f'<img src="data:image/jpg;base64,{item.content}" style="width:128px; height:100px;">'
        items.append(
            {"id": str(item._id),
            "content": content_html,
            "start": datetime.fromtimestamp(item.metadata['timestamp']).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }
        )
    with st.empty():
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
        timeline_placeholder.write(timeline)
        
        initial_message = Message(content=input)
        all_messages = ""
        
        for message in agent_chain.talk([initial_message]):
            progress_status = message.metadata.get('progress', '')
            if "Starting InfoExtractAgent" in progress_status:
                proceed_bar.progress(25, text="agent progress")
            elif "Completed InfoExtractAgent" in progress_status:
                proceed_bar.progress(50, text="agent progress")
            elif "Starting MemoryEditAgent" in progress_status:
                proceed_bar.progress(75, text="agent progress")
            elif "Completed MemoryEditAgent" in progress_status:
                proceed_bar.progress(100, text="agent progress")
                result = RetrievalData.from_json(message.content)

            all_messages += progress_status + "\n"
            text_area_placeholder.text_area("agent status", all_messages, height=200)
            print(progress_status)
            
        last_message = message
        result = RetrievalData.from_json(last_message.content)
        items = []
        print(len(result.items))
        for item in result.items:
            content_html = f'<div>{item.metadata.get("summary", item.content)[:16]}...</div>'
            if item.modality == ModalityType.IMAGE:  # Check if item.content is not empty
                content_html += f'<img src="data:image/jpg;base64,{item.content}" style="width:128px; height:100px;">'
            items.append(
                {"id": str(item._id),
                "content": content_html,
                "start": datetime.fromtimestamp(item.metadata['timestamp']).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            )
        timeline = st_timeline(items, groups=[], height="512px", options={
            "editable": {
                    "add": True,  # add new items by double tapping
                    "updateTime": True,  # drag items horizontally
                    "updateGroup": True,  # drag items from one group to another
                    "remove": True,  # delete an item by tapping the delete button top right
                    "overrideItems": True,  # allow these options to override item.editable
                },
            "selectable": False,
        })
        timeline_placeholder.write(timeline)
    # st.write(timeline)
    