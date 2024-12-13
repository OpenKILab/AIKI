from aiki.aiki import AIKI
import time
import random

ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/wiki/")
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
    'num_texts: 要生成的文本数量',
    'max_length: 每个文本的最大长度',
    'nerate_random_text(min_length=5, max_length=20',
    '可以指定文本的最小和最大长度',
    '如果你想要更改字符集或其他参数，可以：',
    '1. 使用了所有ASCII字母（大小写）、数字、标点符号和空格',
    '每种语言都有多个相似但不完全相同的表达方式',
    '扩充每种语言的模板句子',
]
while True:
    start_time = time.time()
    input = random.choice(sentences)
    print(f"input: {input}")
    result = ak.retrieve(input, num=2)

    # 计算并打印执行时间
    execution_time = time.time() - start_time
    print(f"检索结果：\n{result}")
    print(f"检索耗时：{execution_time:.4f} 秒")

    print(result)