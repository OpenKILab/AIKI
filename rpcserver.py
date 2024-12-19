from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from aiki.aiki import AIKI


server = SimpleXMLRPCServer(
    ("0.0.0.0", 10055),
    requestHandler=SimpleXMLRPCRequestHandler,
    allow_none=True
)

ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/law_industrycorpus2/")

def hello_world():
    return {"message": "Hello World"}

def retrieve(query: str, top_k: int = 1):
    result = ak.retrieve(query, top_k)
    result = [item['content'] for item in result]
    return result

server.register_function(hello_world, "hello_world")
server.register_function(retrieve, "retrieve")

if __name__ == "__main__":
    print("RPC Server running on port 10055...")
    server.serve_forever()