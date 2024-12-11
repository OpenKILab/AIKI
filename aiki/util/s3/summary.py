import os
import ijson
import json
import tiktoken  # Make sure to install this package

def count_content_list_entries_and_tokens(file_path:str, model_name:str = "gpt-4o"):
    count = 0
    total_tokens = 0
    tokenizer = tiktoken.encoding_for_model('gpt-4o')

    with open(file_path, 'r', encoding='utf-8') as file:
        for data in ijson.items(file, '', multiple_values=True):
            if 'content_list' in data:
                count += len(data['content_list'])
                # Calculate tokens for each entry in content_list
                for entry in data['content_list']:
                    tokens = tokenizer.encode(entry['text'])
                    total_tokens += len(tokens)

    return count, total_tokens

if __name__ == "__main__":
    directory_path = '/mnt/hwfile/kilab/leishanzhe/data/wiki'
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory_path, filename)
            total_entries, total_tokens = count_content_list_entries_and_tokens(file_path)
            print(f'File: {filename}')
            print(f'Total entries in content_list: {total_entries}')
            print(f'Total tokens in content_list: {total_tokens}')