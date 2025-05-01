from typing import Annotated

import redis.asyncio as redis
from fastapi import Depends

from services.redis import get_async_redis_session

REDIS_CONN = Annotated[redis.Redis, Depends(get_async_redis_session)]
