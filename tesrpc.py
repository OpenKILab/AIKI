from xmlrpc.client import ServerProxy

client = ServerProxy('http://localhost:10055')

result = client.hello_world()
print(result)  # 输出: {'message': 'Hello World'}

result = client.retrieve("测试查询", 2)
print(result)