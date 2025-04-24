import asyncio

from fastapi import APIRouter

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema

prefix = r"/playlist"
router = APIRouter(prefix=prefix, tags=["playlist"])


@router.post("/recommend/")
async def recommend_playlist(body: PlaylistRequestSchema) -> PlaylistResponseSchema:
    """
    Endpoint to get the recommended playlist for users
    """
    print(body)

    # TODO=> Call the ML Model inference here!!
    await asyncio.sleep(1)

    response_params = {"playlist_id": "123"}
    return PlaylistResponseSchema(**response_params)
