from fastapi import FastAPI

app = FastAPI()
items = []

@app.get("/")
def root():
    return {"hello": "world"}

@app.post("/items")
def create_item(item: str):
    items.append(item)
    return items

@app.get("/items")
def list_item(limit: int=10):
    return items[0:limit]