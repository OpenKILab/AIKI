from abc import ABC, abstractmethod
from aiki.agent.prompts import extract_information_prompt_template, process_memory_prompt_template, merge_memory_prompt_template
from dataclasses import dataclass  
from typing import List
  

@dataclass  
class Message:  
    role: str  
    content: str  


class BaseAgent(ABC):
    @abstractmethod
    def talk(self,message):
        pass

class InfoExtractAgent(BaseAgent):
    def __init__(self, config:dict, extract_model:callable):
        self.extract_model = extract_model
        self.name = 'InfoExtractAgent'
        ...

    def extract_information(self, history:str) -> str:
        ...
        prompt = extract_information_prompt_template.format(history=history)
        information = self.extract_model(prompt)
        return information


    def talk(self, message:List[Message]) -> str:
        history = ""
        for msg in Message:
            history += f"{msg.role}: {msg.content}"
       
        extracted_information = self.extract_information(history)

        return extracted_information




class MemoryEditAgent(BaseAgent):
    def __init__(self, config:dict, memo_database:str, process_model:callable):
        self.memo_database = memo_database 
        self.process_model = process_model
        self.name = 'MemoryEditAgent'
        ...

    def search(self, query, n=3):
        ...
    
    def add(self, id, data):
        ...

    def delete(self, id):
        ...
    
    def edit(self, id, data):
        # merge & replace
        ... 
    
    def merge_memory(self, id, data_1, data_2):
        ... 


    def load_function(self, process_function:str) -> bool:
        try: 
            ...
            return True

        except:
            return False

    def process_memory(self, temp_memory, related_memory) -> str:
        ...
        prompt = process_memory_prompt_template.format(temp_memory=temp_memory,related_memory=related_memory)
        function_call = self.process_model(prompt)
        return function_call
        


    def talk(self, message) -> str:
        temp_memory = message['information']
        related_memory = self.search(self, temp_memory)
        process_function = self.process_memory(temp_memory=temp_memory, related_memory=related_memory)
        while not self.load_function(process_function):
            process_function = self.process_memory(temp_memory=temp_memory, related_memory=related_memory)
        return process_function
        ... 
    ...

    


    