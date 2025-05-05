from fastapi import APIRouter, BackgroundTasks

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema
from ml_model.recommender import get_recommended_playlist
from services.prometheus import INFERENCE_COUNT, INFERENCE_DIVERSITY
from dependencies import REDIS_CONN

prefix = r"/playlist"
router = APIRouter(prefix=prefix, tags=["playlist"])


def update_prometheus(predictions: dict[int, int]):
    for prediction in predictions.values():
        INFERENCE_DIVERSITY.labels(model="playlist_recommender").set(prediction)


@router.post("/recommend/")
async def recommend_playlist(
    body: PlaylistRequestSchema,
    cache: REDIS_CONN,
    background_task: BackgroundTasks,
) -> list[PlaylistResponseSchema]:
    """
    Endpoint to get the recommended playlist for users
    """
    predictions = await get_recommended_playlist(body.user_ids)
    # # Updating the Redis Cache
    await cache.hset(name="playlist", mapping=predictions)
    # Update Prometheus from BackgroundTasks
    background_task.add_task(update_prometheus, predictions)

    INFERENCE_COUNT.inc(amount=len(predictions))

    return [
        PlaylistResponseSchema(user_id=user_id, playlist_id=playlist_id)
        for user_id, playlist_id in predictions.items()
    ]
    # return PlaylistResponseSchema(**response_params)
