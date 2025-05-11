from contextlib import asynccontextmanager

from fastapi import FastAPI

from redis_client import get_async_redis_session, redis_conn_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model here
    print("Starting Server")

    print("Starting Redis")
    await (await get_async_redis_session()).ping()

    # Load the ML model
    # model.load_model(settings.MODEL_PATH)

    yield

    # Close Redis Connection Pool
    print("Closing Redis Connection Pool")
    await redis_conn_pool.aclose()
    print("Closing Server")
