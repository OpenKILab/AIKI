# main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from search import generate_info
import subprocess

# 必须关闭代理
subprocess.run("proxy_off", shell=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

@app.post("/get_info/")
async def server_search(request: Request):
    json_data = await request.json()
    question = json_data.get("question")
    info, webs = generate_info(question)
    
    return JSONResponse(content={"info": info, "webs": webs})
    
    