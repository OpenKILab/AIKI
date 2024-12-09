import json

def count_content_list_entries(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if 'content_list' in data:
                count += len(data['content_list'])
    return count

file_path = 'part-6656fc2cb915-005243.jsonl'
total_entries = count_content_list_entries(file_path)
print(f'Total entries in content_list: {total_entries}')