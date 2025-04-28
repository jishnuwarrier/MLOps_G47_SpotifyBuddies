from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.redis import redis_conn_pool, get_async_redis_session
from config import settings
from ml_model.model import model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model here
    print("Starting Server")

    print("Starting Redis")
    await (await get_async_redis_session()).ping()

    # Load the ML model
    model.load_model(settings.MODEL_PATH)

    yield

    # Close Redis Connection Pool
    print("Closing Redis Connection Pool")
    await redis_conn_pool.aclose()
    print("Closing Server")
