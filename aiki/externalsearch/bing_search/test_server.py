# test_get_info.py

import requests  # 使用 requests 库替代 httpx

problem_list = [
    "人工智能医疗诊断系统的伦理与法律边界",
    "比特币矿场对区域性水资源的影响",
    "星链卫星网络的天文学干扰与经济替代方案",
    "垂直农业的能源效率悖论",
    "脑机接口技术的社会阶层分化风险",
    "氢能源船舶的供应链碳足迹追踪",
    "短视频算法与青少年多巴胺分泌模式的关联研究",
    "3D打印建筑对发展中国家劳动力市场的冲击",
    "元宇宙虚拟土地交易的金融监管漏洞",
    "新冠疫苗冷链运输的极地航线优化"
]

def test_get_info(question):
    url = "http://localhost:10033/info/"
    data = {"question": question, 'image': ""}  # 这里是 JSON 格式的字典
    headers = {"Content-Type": "application/json"}  # 定义请求头
    
    response = requests.post(url, json=data, headers=headers)  # 使用 requests.post 发送 JSON 数据和请求头
    
    if response.status_code == 200:
        print("Response JSON:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    for question in problem_list:
        test_get_info(question)  # 测试每个问题