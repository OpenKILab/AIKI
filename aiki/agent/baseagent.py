from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from aiki.agent.prompts import extract_information_prompt_template, time_inference_prompt_template, memory_selection_prompt_template
from typing import Generator, List
import json
from datetime import datetime
from datetime import datetime
from aiki.modal.retrieval_data import RetrievalData
from aiki.bridge.rag_agent_bridge import RAGAgentBridge, RetrievalData
from aiki.multimodal.base import ModalityType
from aiki.multimodal.text import TextModalityData
from aiki.multimodal.image import ImageModalityData
from bson import ObjectId
from aiki.proxy.query import query_llm

import time

class AgentAction(Enum):
    QUERY = 'Query'
    ADD = 'Add'
    REPLACE = 'Replace'
    DELETE = 'Delete'


@dataclass  
class Message:  
    role: str = "user" 
    type: str = "text"
    content: str = ""
    action: AgentAction = AgentAction.QUERY
    metadata: dict = field(default_factory=dict)
    new_memory: RetrievalData = None




def parse_json(text: str) -> dict:
    # 查找字符串中的 JSON 块
    if "json" in text:  
        start = text.find("```json")
        end = text.find("```", start + 7)
        # 如果找到了 JSON 块
        if start != -1 and end != -1:
            json_string = text[start + 7: end]
            try:
                # 解析 JSON 字符串
                json_data = json.loads(json_string)
                #valid = check_selector_response(json_data)
                return json_data
            except:
                print(f"error: parse \"json\" error!\n")
                print(f"json_string: {json_string}\n\n")
                pass
    elif "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if start != -1 and end != -1:
            json_string = text[start + 3: end]
            
            try:
                # 解析 JSON 字符串
                json_data = json.loads(json_string)
                return json_data
            except:
                print(f"error: parse json ``` error!\n")
                print(f"json_string: {json_string}\n\n")
                pass
    else:
        start =  text.find("{")
        end = text.find("}", start + 1)
        if start != -1:
            json_string = text[start: end + 1]
            try:
                # 解析 JSON 字符串
                json_data = json.loads(json_string)
                return json_data
            except:
                print(f"error: parse json error!\n")
                print(f"json_string: {json_string}\n\n")
                pass
    return {}

class BaseAgent(ABC):
    @abstractmethod
    def talk(self,message):
        pass

class InfoExtractAgent(BaseAgent):
    def __init__(self, config:dict, extract_model:callable):
        self.extract_model = extract_model
        self.name = 'InfoExtractAgent'
        ...

    def extract_information(self, context:str) -> str:
        ...
        prompt = extract_information_prompt_template.format(user_input=context)
        information = self.extract_model(prompt)
        return information

    def get_time(self, vague_time:str) -> str:
        current_time = datetime.now()
        format_string = "%Y-%m-%d %H:%M:%S"
        current_time_str = current_time.strftime(format_string)
        prompt = time_inference_prompt_template.format(vague_time_description=vague_time,current_precise_time=current_time_str)
        time_answer = self.extract_model(prompt)
        return time_answer
    
    def str_to_timestamp(self, time_str:str) -> float:
        format_str = "%Y-%m-%d %H:%M:%S"
        dt_object = datetime.strptime(time_str, format_str)
        timestamp = dt_object.timestamp()
        return timestamp



    def talk(self, message:List[Message]) -> Message:
        context = ""
        for msg in message:
            if msg.type == 'text':
                context = msg.content
            elif msg.type == 'image':
                context = msg.content
                
        extracted_information = self.extract_information(context)
        informatioin_dict = parse_json(extracted_information)
        target_memory = informatioin_dict['User Memory']
        vague_time = target_memory[1]
        time_answer = self.get_time(vague_time)
        time_dict = parse_json(time_answer)
        query = target_memory[0] + target_memory[2]
        time_dict['start_time'] = int(self.str_to_timestamp(time_dict['start_time']))
        time_dict['end_time'] = int(self.str_to_timestamp(time_dict['end_time']))
        print(time_dict['start_time'])
        print(time_dict['end_time'])
        new_message = Message()
        new_message.metadata = time_dict
        new_message.content = query 
        new_message.action = AgentAction[informatioin_dict['User Intent'].upper()]
        new_message.new_memory = RetrievalData(
            items=[
                TextModalityData(
                    content=context,
                    _id=ObjectId(),
                    metadata={"timestamp": int(datetime.now().timestamp())}
                )
            ]
        )
        return new_message




class MemoryEditAgent(BaseAgent):
    def __init__(self, config:dict, process_model:callable):
        self.process_model = process_model
        self.name = 'MemoryEditAgent'
        self.memoryfunction = {AgentAction.ADD:self.add, AgentAction.QUERY:self.search, AgentAction.DELETE:self.delete, AgentAction.REPLACE:self.replace}
        self.rag = RAGAgentBridge(name="flicker8k_xiaobu")
        ...

    def search(self, message:Message) -> Message:
        retrieval_data = RetrievalData(items=[
        TextModalityData(
            content= message.content,
            _id = ObjectId(),
            metadata=message.metadata
        )
        ])
    
        results = self.rag.query(retrieval_data)
        """
        memory_pool = ""
        memory_dict = {}
        for i in range(len(results.items)):
            temp_memory = ""
            if results.items[i].modality == ModalityType.TEXT:
                temp_memory =  results.items[i].content
            elif results.items[i].modality == ModalityType.IMAGE:
                temp_memory = results.items[i].metadata['summary']
            memory_pool += f"{str(i + 1)}. id: {results.items[i]._id}\nmemory: {temp_memory}\n"
            memory_dict[results.items[i]._id] = results.items[i].to_json()
        prompt = memory_selection_prompt_template.format(target_memory=message.content, memory_pool=memory_pool)
        select_result = parse_json(self.process_model(prompt))
        selected_ids = select_result['selected_ids']
        final_result = []
        for memory_id in selected_ids:
            final_result.append(memory_dict[memory_id])
        message.content = str(final_result)
        """
        message = Message(
            content = results.to_json()
        )
        return message
    
    def add(self, message:Message) -> Message:
        add_data = message.new_memory
        self.rag.add(add_data)
        message.content = message.new_memory.to_json()
        return message

    def delete(self, message:Message) -> Message:
        ...
    
    def replace(self, message:Message) -> Message:
        # merge & replace
        ... 
    



    def talk(self, message:Message) -> Message:
        action = message.action
        print(action)
        return self.memoryfunction[action](message)


class AgentChain():
    def __init__(self, config:dict = None, core_model:callable = None):
        self.core_model = core_model if core_model is not None else query_llm
        self.extractagent = InfoExtractAgent(config, self.core_model)
        self.editagent = MemoryEditAgent(config, self.core_model)
        ...
    
    def talk(self, message:List[Message]) -> Generator[Message, None, None]:
        # Update message metadata and yield progress for the first agent
        message[0].metadata['progress'] = "Starting InfoExtractAgent"
        yield message[0]
        time.sleep(1)
        # extracted_message = self.extractagent.talk(message)
        extracted_message = Message()
        extracted_message.metadata['progress'] = "Completed InfoExtractAgent"
        yield extracted_message
        time.sleep(1)
        # Update message metadata and yield progress for the second agent
        extracted_message.metadata['progress'] = "Starting MemoryEditAgent"
        yield extracted_message
        # final_message = self.editagent.talk(extracted_message)
        time.sleep(1)
        final_message = Message()
        final_message.metadata['progress'] = "Completed MemoryEditAgent"
        yield final_message
    
if __name__ == "__main__":
    agent_chain = AgentChain()
    result = agent_chain.talk([Message(
        content = input
    )])
    # result = RetrievalData.from_json(result.content)