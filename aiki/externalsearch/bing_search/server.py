# main.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from search import (
    generate_info,
    jina_search,
    generate_info_simple,
)
import subprocess

# Set up logging
logging.basicConfig(filename='execution_times.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 必须关闭代理
subprocess.run("proxy_off", shell=True)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

@app.post("/info/")
async def server_search_simple(request: Request):
    json_data = await request.json()
    question = json_data.get("question")
    info, webs = generate_info_simple(question)
    
    return JSONResponse(content={"info": info, "webs": webs})