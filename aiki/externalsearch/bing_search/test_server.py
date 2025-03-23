# test_get_info.py

import requests  # 使用 requests 库替代 httpx

def test_get_info():
    url = "http://localhost:8000/get_info/"
    question = {"question": "哪种火龙果比较好吃"}  # 这里是 JSON 格式的字典
    headers = {"Content-Type": "application/json"}  # 定义请求头
    
    response = requests.post(url, json=question, headers=headers)  # 使用 requests.post 发送 JSON 数据和请求头
    
    if response.status_code == 200:
        print("Response JSON:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_get_info()