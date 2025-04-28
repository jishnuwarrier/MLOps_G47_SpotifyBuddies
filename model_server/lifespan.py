from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.redis import redis_conn_pool, get_async_redis_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model here
    print("Starting Server")
    await (await get_async_redis_session()).ping()
    yield

    # Close Redis Connection Pool
    print("Closing Redis Connection Pool")
    await redis_conn_pool.aclose()
    print("Closing Server")
