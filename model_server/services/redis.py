from redis import asyncio as redis

from config import settings


redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

redis_conn_pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)


async def get_async_redis_session() -> redis.Redis:
    return redis.Redis(connection_pool=redis_conn_pool)
