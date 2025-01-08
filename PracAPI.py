from typing import Optional

from fastapi import FastAPI
import pydantic as pyd
app = FastAPI()

class post(pyd.BaseModel):
    title: str
    content: str
    published: bool = True
    rating: Optional[int] = None

@app.post("/post")
async def create_post(post: post):
    return post

