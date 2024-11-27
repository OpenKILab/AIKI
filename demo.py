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
from proceed import ProceedComponent, proceed_component

model = SentenceTransformer('lier007/xiaobu-embedding-v2')

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
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
# 在右边侧边栏放置 my_bar, text_area_placeholder 和 button_placeholder
with st.sidebar:
    my_bar = st.progress(1)
    text_area_placeholder = st.empty()
    button_placeholder = st.empty()

    st.write(text_area_placeholder.text_area("agent status", "Initial", height=200))

    if 'talk_completed' not in st.session_state:
        st.session_state.talk_completed = False

    with button_placeholder:
        proceed_button = st.button("Proceed", disabled=True, key="proceed_button_initial")


# 更新右边一列的内容
# with right_column:
#     my_bar = st.progress(1)
#     initial_message = Message(content="content")
#     all_messages = ""

#     text_area_placeholder = st.empty()
#     button_placeholder = st.empty()
    
#     st.write(text_area_placeholder.text_area("agent status", "Initial", height=200))
    
#     if 'talk_completed' not in st.session_state:
#         st.session_state.talk_completed = False

#     with button_placeholder:
#         proceed_button = st.button("Proceed", disabled=True, key="proceed_button_initial")

#     last_message = None
    
    # for message in agent_chain.talk([initial_message]):
    #     progress_status = message.metadata.get('progress', '')
    #     if "Starting InfoExtractAgent" in progress_status:
    #         my_bar.progress(25)
    #     elif "Completed InfoExtractAgent" in progress_status:
    #         my_bar.progress(50)
    #     elif "Starting MemoryEditAgent" in progress_status:
    #         my_bar.progress(75)
    #     elif "Completed MemoryEditAgent" in progress_status:
    #         my_bar.progress(100)

    #     all_messages += progress_status + "\n"
    #     text_area_placeholder.text_area("agent status", all_messages, height=200)
    #     print(progress_status)
        
    #     last_message = message

    # st.session_state.talk_completed = True

    # with button_placeholder:
    #     proceed_button = st.button("Proceed", disabled=not st.session_state.talk_completed, key="proceed_button_final")



# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

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
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            # content="我儿子玩雪玩的很开心",
            content=input,
            _id=ObjectId(),
            metadata={}
        )
    ])
    result = dense_retriever.search(retrieval_data, num=3)
    '''
    agent_chain = AgentChain()
    result = agent_chain.talk([Message(
        content = input
    )])
    result = RetrievalData.from_json(result.content)
    '''
    # 根据检索到的结果回复
    prompt = (f"Here are some images/texts related to the query:")
    for item in result.items:
        prompt += f"\nnew images/texts : 时间: {item.metadata['timestamp']} 内容：{item.metadata.get('summary', item.content)}"
    prompt += f"\n根据这些图片来回复用户的问题：{input}。"
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    # 输出timeline
    items = []
    for item in result.items:
        # print(item.metadata)
        content_html = f'<div>{item.metadata.get("summary", item.content)[:16]}...</div>'
        if item.modality == ModalityType.IMAGE:  # Check if item.content is not empty
            content_html += f'<img src="data:image/jpg;base64,{item.content}" style="width:128px; height:100px;">'
        items.append(
            {"id": str(item._id),
            "content": content_html,
            "start": datetime.fromtimestamp(item.metadata['timestamp']).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }
        )
    timeline = st_timeline(items, groups=[], options={}, height="512px")
    
    
    initial_message = Message(content="content")
    all_messages = ""

    for message in agent_chain.talk([initial_message]):
        progress_status = message.metadata.get('progress', '')
        if "Starting InfoExtractAgent" in progress_status:
            my_bar.progress(25)
        elif "Completed InfoExtractAgent" in progress_status:
            my_bar.progress(50)
        elif "Starting MemoryEditAgent" in progress_status:
            my_bar.progress(75)
        elif "Completed MemoryEditAgent" in progress_status:
            my_bar.progress(100)

        all_messages += progress_status + "\n"
        text_area_placeholder.text_area("agent status", all_messages, height=200)
        print(progress_status)
        
        last_message = message

st.session_state.talk_completed = True

with button_placeholder:
    proceed_button = st.button("Proceed", disabled=not st.session_state.talk_completed, key="proceed_button_final")
    
    # st.write(timeline)
    