from fastapi import APIRouter

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema
from ml_model.recommender import get_recommended_playlist
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
    response_params = {"playlist_id": playlist_id}

    # Updating the Redis Cache
    await cache.set(body.user_id, playlist_id)

    return PlaylistResponseSchema(**response_params)
