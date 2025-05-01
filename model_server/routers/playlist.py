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
) -> PlaylistResponseSchema:
    """
    Endpoint to get the recommended playlist for users
    """
    playlist_id = await get_recommended_playlist(body.user_id)
    INFERENCE_COUNT.inc()
    INFERENCE_COUNTER_OUPUT.labels(output_label=str(playlist_id)).inc()
    response_params = {"playlist_id": playlist_id}

    # Updating the Redis Cache
    with REDIS_TO_TIME.time():
        await cache.set(body.user_id, playlist_id)

    return PlaylistResponseSchema(**response_params)
