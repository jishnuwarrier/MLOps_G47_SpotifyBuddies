import json
import asyncio

import psycopg2
import sql
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

if settings.LOCAL:
    templates = Jinja2Templates(directory="templates")
else:
    templates = Jinja2Templates(directory="app/templates")

# Connect to postgres
conn = psycopg2.connect(
    dbname=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
)

with conn as cursor:
    cursor.execute(open("schema.sql", "r").read())


class UserResponse(BaseModel):
    playlist_ids: list[int]


class FeedbackRequestSchema(BaseModel):
    user_id: int
    playlist_id: int


@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running or not
    """
    await asyncio.sleep(1)
    return {"Status": "Server Running"}


@app.get("/playlist")
async def get_playlists(user_id: int, DB=Depends(get_async_redis_session)):
    playlist = await DB.hget("playlist", user_id)

    if settings.DEBUG:
        return json.loads(playlist) * 5

    return json.loads(playlist)


@app.post("/feedback")
def send_feedback(req: FeedbackRequestSchema):
    conn.execute(
        sql.SQL("insert into {} values (%s, %s)").format(
            sql.Identifier("user_feedback ")
        ),
        [req.user_id, req.playlist_id],
    )

    return {"msg": "Feedback Sent"}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
