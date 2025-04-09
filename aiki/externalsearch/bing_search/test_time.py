# 定义文件路径
file_path = "/fs-computility/ai-shen/leishanzhe/repo/AIKI/aiki/externalsearch/bing_search/execution_times.log"

# 初始化一个列表来存储时间
times = []

# 读取文件并提取时间
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # 反向遍历文件的最后十行
    for line in reversed(lines):
        if "Total time taken for generate_info_simple" in line:
            # 提取时间并转换为浮点数
            time_taken = float(line.split(': ')[-1].strip().split(' ')[0])
            times.append(time_taken)
            # 只保留最后十个时间
            if len(times) == 10:
                break

# 计算总时间
total_time = sum(times)

# 输出总时间
print(f"Total time for the last ten entries: {total_time} seconds")