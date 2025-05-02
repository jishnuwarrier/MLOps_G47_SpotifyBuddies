import asyncio
import sys

from .model import pool, make_prediction
from services.prometheus import INFERENCE_TIME

semaphore = asyncio.Semaphore(10)


# TODO => Improve it after the stable model is created
@INFERENCE_TIME.time()
async def get_recommended_playlist(user_id: int) -> int:
    loop = asyncio.get_event_loop()

    # result = await loop.run_in_executor(pool, model.predict, user_id)
    if sys.platform == "win32":
        async with semaphore:
            result = await loop.run_in_executor(pool, make_prediction, user_id)
    else:
        result = await loop.run_in_executor(pool, make_prediction, user_id)
    return result
