import asyncio

from fastapi import APIRouter

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema
from ml_model.recommender import get_recommended_playlist
from services.prometheus import INFERENCE_COUNT, REDIS_TO_TIME, INFERENCE_COUNTER_OUPUT
from dependencies import REDIS_CONN

prefix = r"/playlist"
router = APIRouter(prefix=prefix, tags=["playlist"])


@router.post("/recommend/")
async def recommend_playlist(
    body: PlaylistRequestSchema, cache: REDIS_CONN
) -> list[PlaylistResponseSchema]:
    """
    Endpoint to get the recommended playlist for users
    """
    playlist_ids = await get_recommended_playlist(body.user_ids)
    # # Updating the Redis Cache
    with REDIS_TO_TIME.time():
        tasks = []
        for user_id, playlist_id in zip(body.user_ids, playlist_ids):
            INFERENCE_COUNTER_OUPUT.labels(output_label=str(playlist_id)).inc()
            task = cache.set(user_id, playlist_id)
            tasks.append(task)
        await asyncio.gather(*tasks)
        # await cache.set(body.user_id, playlist_id)
    INFERENCE_COUNT.inc()
    return [
        PlaylistResponseSchema(user_id=user_id, playlist_id=playlist_id)
        for user_id, playlist_id in zip(body.user_ids, playlist_ids)
    ]
    # return PlaylistResponseSchema(**response_params)
