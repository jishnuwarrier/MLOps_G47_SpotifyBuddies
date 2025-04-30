import asyncio

from .model import pool, make_prediction


# TODO => Improve it after the stable model is created
async def get_recommended_playlist(user_id: int) -> int:
    loop = asyncio.get_event_loop()

    # result = await loop.run_in_executor(pool, model.predict, user_id)
    result = await loop.run_in_executor(pool, make_prediction, user_id)
    return result
