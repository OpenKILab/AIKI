extract_information_prompt_template = """ 
[Instruction]

Your task is to extract information from a single user input. Specifically, you will identify and categorize the information into two parts: User Intent and User Memory (in the form of triplets). This process involves understanding the context, intent, and content of the user input to accurately capture the required information.

[Requirements]

User Intent: ##Identify the user's intent from the input. ##The possible intents are: - Query: The user is asking for information. - Add: The user is adding new information. - Replace: The user is replacing existing information. - Delete: The user is requesting to delete information.

User Memory: ##Identify triplet of the form (Role, Time, Activity), extract one triplet per input. ##Roles refers to the subject of this memory. This "role" does not necessarily correspond to the subject of the input sentence, but rather to the object of ownership or association for the target memory being referred to in the input. ##Time may include specific dates, times, or general time frames (e.g., "today", "last week"). If no specific time is mentioned, default to 'all'. ##Activities may include tasks, actions, or events (e.g., "eating at restaurants", "recording a family dinner"). ##Ensure that each triplet clearly represents a the target memory for each input.

Context Understanding: ##Carefully read the user input to understand the context. ##Use your understanding of language and context to infer information when it is not explicitly stated.

Accuracy and Precision: ##Ensure that the extracted information is accurate and precisely represents the user input. ##Avoid including irrelevant or redundant information. Output Language: ##You should respond in tha same language as the user input [Examples] ========================================Example 1========================================

Example 1: User-Defined Information

User Input: "Where did I eat last week?"

Extracted Information:

```json
{{ 
    "User Intent": "Query", 
    "User Memory": [("I", "last week", "eating at restaurants")]
}}
```

========================================Example 2========================================

Example 2: User-Defined Information

User Input: "I had some unpleasant moments during the family dinner two days ago. Please delete the related records."

Extracted Information:

```json
{{ 
    "User Intent": "Delete", 
    "User Memory": [("I", "two days ago", "having a family dinner unpleasantly")] 
}}
```

========================================Example 3========================================

Example 3: User-Defined Information

User Input: "My son took some new photos today."

Extracted Information:

```json
{{ 
    "User Intent": "Add", 
    "User Memory": [("My son", "today", "taking new photos")] 
}}
```

========================================Example 4========================================

Example 4: User-Defined Information

User Input: "I want to replace the old family photos with the new ones I just received."

Extracted Information:

```json
{{ 
    "User Intent": "Replace", 
    "User Memory": [("I", "all", "taking photos with family in the past")] 
}}
```

========================================Example End======================================== By following these instructions and requirements, you will be able to accurately extract and categorize information from a single user input. Remember to focus on understanding the context and intent of the user input to ensure that the extracted information is accurate and relevant.

[Your Task] User Input: 
{user_input}
Extracted Information:
"""

time_inference_prompt_template = """ 
[Instruction]

Given a vague time description and the current precise time, determine the corresponding time range and return it in JSON format. For vague descriptions like "recently" or "a few days ago," consider them as within the past week. If vague description is 'all', it is regarded as the time period from 100 years ago until today.

[Requirements]
-Vague Time Description: A vague time period (e.g., "yesterday," "last week," "recently").
-Current Precise Time: The exact current time (e.g., "2024-11-13 17:45:51").
-Output: JSON with "start_time" and "end_time" keys.

[Examples] ========================================Example 1========================================

Input:
##Vague Time Description: "yesterday"
##-Current Precise Time: "2024-11-13 17:45:51"

Output:

```json
{{
    "start_time": "2024-11-12 00:00:00",
    "end_time": "2024-11-12 23:59:59"
}}
```

========================================Example 2========================================

Input:
##Vague Time Description: "last week"
##Current Precise Time: "2024-11-13 17:45:51"

Output:

```json
{
    "start_time": "2024-11-06 00:00:00",
    "end_time": "2024-11-12 23:59:59"
}
```

========================================Example 3========================================

Input:
##Vague Time Description: "recently"
##Current Precise Time: "2024-11-13 17:45:51"

Output:

```json
{
    "start_time": "2024-11-06 00:00:00",
    "end_time": "2024-11-13 17:45:51"
}
```

========================================Example End========================================

[Your Task] Input:
##Vague Time Description: {vague_time_description}
##Current Precise Time: {current_precise_time}

Output:
"""

memory_selection_prompt_template = """ 
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