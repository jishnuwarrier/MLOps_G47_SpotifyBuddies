import json
import asyncio
import time
from pathlib import Path

import psycopg2
from psycopg2 import sql, OperationalError
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
conn_str = "dbname=client user=postgres password=1234 host=postgres"
# conn_str = "postgresql+psycopg2://postgres:1234@postgres/client"
if settings.LOCAL:
    templates = Jinja2Templates(directory="templates")
else:
    templates = Jinja2Templates(directory="app/templates")
max_retries = 10
retry_delay = 3  # seconds

for attempt in range(max_retries):
    try:
        # conn = psycopg2.connect(conn_str)
        conn = psycopg2.connect(
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
        )
        break  # Connection successful
    except OperationalError as e:
        if "the database system is starting up" in str(e):
            print(f"Database is starting up. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            raise  # Unexpected error, re-raise
else:
    raise Exception("Could not connect to the database after several retries.")


sql_path = Path(__file__).parent / "schema.sql"
with open(sql_path, "r") as file:
    sql_script = file.read()

with conn:
    with conn.cursor() as cur:
        cur.execute(sql_script)


class UserResponse(BaseModel):
    playlist_ids: list[int]


class FeedbackRequestSchema(BaseModel):
    user_id: int
    playlist_id: int
    score: int


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
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("insert into {} values (%s, %s, %s)").format(
                    sql.Identifier("user_feedback")
                ),
                [req.user_id, req.playlist_id, req.score],
            )

    return {"msg": "Feedback Sent"}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
