import time
import streamlit as st
from aiki.agent.baseagent import AgentChain, Message

class ProceedComponent():
    def __init__(self, agent_chain: AgentChain):
        self.agent_chain = agent_chain
    
    def input(self, content: str) -> Message:
        my_bar = st.progress(1)
        initial_message = Message(content=content)
        all_messages = ""

        text_area_placeholder = st.empty()
        button_placeholder = st.empty()
        
        st.write(text_area_placeholder.text_area("agent status", "Initial", height=200))
        
        if 'talk_completed' not in st.session_state:
            st.session_state.talk_completed = False

        with button_placeholder:
            proceed_button = st.button("Proceed", disabled=True, key="proceed_button_initial")

        last_message = None
        
        for message in self.agent_chain.talk([initial_message]):
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

        return last_message
        
def proceed_component(agent_chain: AgentChain, content: str) -> Message:
    my_bar = st.progress(1)
    initial_message = Message(content=content)
    all_messages = ""

    text_area_placeholder = st.empty()
    button_placeholder = st.empty()
    
    st.write(text_area_placeholder.text_area("agent status", "Initial", height=200))
    
    if 'talk_completed' not in st.session_state:
        st.session_state.talk_completed = False

    with button_placeholder:
        proceed_button = st.button("Proceed", disabled=True, key="proceed_button_initial")

    last_message = None
    
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

    return last_message

# Example usage
if __name__ == "__main__":
    agent_chain = AgentChain()
    user_input = st.text_input("Enter your content:")
    if st.button("Run Component"):
        result = proceed_component(agent_chain, user_input)
        if result:
            st.write("Last message:", result)
        else:
            st.write("No message returned.")