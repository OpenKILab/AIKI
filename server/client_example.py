import time
from xmlrpc.client import ServerProxy

client = ServerProxy('http://10.140.0.153:10055')

result = client.hello_world()
print(result)  # 输出: {'message': 'Hello World'}

start_time = time.time()
result = client.retrieve("查询语句：电动汽车销售代理合同纠纷 法律判决条件及依据", 1)
end_time = time.time()
execution_time = end_time - start_time

print(f"查询结果: {result}")
print(f"查询耗时: {execution_time:.2f} 秒")
print(result)