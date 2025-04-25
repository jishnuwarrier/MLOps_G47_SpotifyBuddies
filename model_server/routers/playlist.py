from fastapi import APIRouter

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema
from ml_model.recommender import get_recommended_playlist

prefix = r"/playlist"
router = APIRouter(prefix=prefix, tags=["playlist"])


@router.post("/recommend/")
async def recommend_playlist(body: PlaylistRequestSchema) -> PlaylistResponseSchema:
    """
    Endpoint to get the recommended playlist for users
    """
    response_params = {"playlist_id": get_recommended_playlist(body.user_id)}
    return PlaylistResponseSchema(**response_params)
