import json
import asyncio

from fastapi import FastAPI, Depends, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from lifespan import lifespan
from redis_client import get_async_redis_session
from config import settings

app = FastAPI(
    title="Client App",
    description="This service to expose the API endpoint to the client would interact with",
    version="0.0.1",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


class UserResponse(BaseModel):
    playlist_ids: list[int]


@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running or not
    """
    await asyncio.sleep(1)
    print(settings)
    return {"Status": "Server Running"}


@app.get("/playlist")
async def get_playlists(user_id: int, DB=Depends(get_async_redis_session)):
    playlist = await DB.hget("playlist", user_id)

    if settings.DEBUG:
        return json.loads(playlist) * 5

    return json.loads(playlist)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
