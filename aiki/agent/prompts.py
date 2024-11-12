extract_information_prompt_template = """ 
[Instruction]

Your task is to extract information from multi-turn conversations. Specifically, you will identify and categorize the information into two types: User-Defined Information (in the form of triplets) and Knowledge Information (in the form of title-content pairs). This process involves understanding the context, intent, and content of each conversation turn to accurately capture the required information.

[Requirements]

****User-Defined Information****:
    ##Identify triplets of the form (Character, Attribute, Value).
    ##Attributes may include preferences (e.g., favorite movie genres), current needs (e.g., looking for programming-related projects), tasks (e.g., investigating a murder case), defined formats (e.g., email format requirements), or roles (e.g., role of the conversation assistant).
    ##Ensure that each triplet clearly represents a user-specific definition or attribute.

****Knowledge Information****:
    ##Extract knowledge snippets from question-answer pairs.
    ##Summarize the main idea of the snippet and create a title that reflects this idea.
    ##Store the information in the format of "Title: Content".
    ##Ensure that the title accurately represents the content and is concise.

****Context Understanding****:
    ##Carefully read each conversation turn to understand the context.
    ##Pay attention to the flow of the conversation and how information is presented.
    ##Use your understanding of language and context to infer information when it is not explicitly stated.

****Accuracy and Precision****:
    ##Ensure that the extracted information is accurate and precisely represents the conversation.
    ##Avoid including irrelevant or redundant information.

[Examples]
========================================Example 1========================================

Example 1: User-Defined Information

Conversation:
User: "I'm looking for a project related to programming."
Assistant: "Great! Do you have any specific requirements?"
User: "No, just something that allows me to practice my coding skills."
Assistant: "OK, I will help you to find a suitable project. Do you have any other requests?"
User: "I previously shared some information about my previous job with you, which wasn't quite appropriate. Please forget about the related information I shared earlier."

Extracted Information:
```json
{{
    'User-Defined': [(User, Current Need, Looking for a project that is related to programming and can be used to practice coding skills.),(User, Memories that are hoped to be forgotten, Previous information about the user's job)],
    'Knowledge': []
}}
```

========================================Example 2========================================

Example 2: Knowledge Information

Conversation:
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

Extracted Information:
```json
{{
    'User-Defined': [],
    'Knowledge': [('Title: Capital of France','Content: The capital of France is Paris.')]
}}
```

========================================Example 3========================================

Example 3: Combined User-Defined and Knowledge Information

Conversation:
User: "You are a detective and you are currently investigating a murder case. I'll give you information and work with you on the mission."
Assistant: "That sounds serious. Do you have any leads?"
User: "Yes, we found a bloody knife at the crime scene. It's likely the murder weapon."

Extracted Information:
```json
{{
    'User-Defined': [(Assistant, Role, Detective), (Assistant, Current Task, Investigating a murder case)],
    'Knowledge': [('Title: Murder Weapon Found','Content: A bloody knife was found at the crime scene, likely the murder weapon.')]
}}
```

========================================Example End========================================
By following these instructions and requirements, you will be able to accurately extract and categorize information from multi-turn conversations. Remember to focus on understanding the context and intent of each conversation turn to ensure that the extracted information is accurate and relevant.

[Your Task]
Conversation:
{history}

Extracted Information:
"""



memory_edit_prompt_template = """ 
[Instruction]
Analyze the new memory and the related old memories. Determine the appropriate action for the new memory:
    **Add**: Directly add the new memory to the memory bank.
    **Replace**: Replace a specific old memory (identified by its ID) with the new memory.
    **Delete**: Delete specific old memories (identified by their IDs) based on the new memory's content or user's intention.
    **Merge**: Merge the new memory with a specific old memory (identified by its ID) to create a new, combined memory.

[Function Call Format]
```json
{{
    'action': '<action>',
    'memory_id_list': [<memory_id_if_needed>]
}}
```

[Examples]
========================================Example 1========================================

New memory: (User, Favorite English Movie, Harry Potter)

Related old memories:
1. id: 0
memory: (User, Favorite English movie, The Twilight Saga)
2. id: 1
memory: (User, Favorite Chinese movie, King of Comedy)
3. id: 2
memory: (User, Absolutely hated movie, Transformers: The Last Knight)

Action:
```json
{{
    'action': 'Replace',
    'memory_id_list': [0]
}}
```

========================================Example 2========================================
New memory: (User, Favorite English Movie, Harry Potter)

Related old memories:
1. id: 0
memory: (User, Favorite English movie, The Twilight Saga)
2. id: 1
memory: (User, Favorite Chinese movie, King of Comedy)
3. id: 2
memory: (User, Absolutely hated movie, Transformers: The Last Knight)

Action:
```json
{{
    'action': 'Replace',
    'memory_id_list': [0]
}}
```
"""