from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def talk(self,message):
        pass

class InfoExtractAgent(BaseAgent):
    def __init__(self, config, extract_model):
        self.extract_model = extract_model
        self.name = 'InfoExtractAgent'
        ...

    def extract_information(self, contents):
        ...
        prompt = ...
        information = self.extract_model(prompt)
        ...

    def talk(self, message):
        if message['send_to'] != self.name:
            return 
        new_input = message['new_history']
        extracted_information = self.extract_information(new_input)
        message['information'] = extracted_information
        message['send_to'] = 'MemoryEditAgent'
        ... 



class MemoryEditAgent(BaseAgent):
    def __init__(self, config, memo_database, process_model):
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

    def process_memory(self, temp_memory, related_memory):
        ...
        prompt = ...
        function_call = self.process_model(prompt)
        ...

    def merge_memory(self, initial_data, new_data):
        ... 

    def talk(self, message):
        if message['send_to'] != self.name:
            return 
        temp_memory = message['information']
        related_memory = self.search(self, temp_memory)
        process_function = self.process_memory(temp_memory=temp_memory, related_memory=related_memory)
        message['send_to'] = 'System'
        ... 
    ...

    


    