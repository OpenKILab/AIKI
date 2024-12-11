import ijson

def count_content_list_entries(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        # Use ijson to parse the file incrementally
        for data in ijson.items(file, '', multiple_values=True):
            if 'content_list' in data:
                count += len(data['content_list'])
                for content in data['content_list']:
                    print(content)
    return count

file_path = '/mnt/hwfile/kilab/leishanzhe/data/ey/part-6656fc2cb915-005243.jsonl'
total_entries = count_content_list_entries(file_path)
print(f'Total entries in content_list: {total_entries}')