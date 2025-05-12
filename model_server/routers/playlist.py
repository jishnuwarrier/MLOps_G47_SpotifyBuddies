import json

from fastapi import APIRouter, BackgroundTasks

from schemas.playlist import PlaylistRequestSchema, PlaylistResponseSchema
from ml_model.recommender import get_recommended_playlist
from services.prometheus import INFERENCE_COUNT, INFERENCE_DIVERSITY, COLD_USER
from dependencies import REDIS_CONN
from services.rabbitmq import send_message_to_queue

prefix = r"/playlist"
router = APIRouter(prefix=prefix, tags=["playlist"])


def update_prometheus(predictions: dict[int, list[int]]):
    for predictions in predictions.values():
        [INFERENCE_DIVERSITY.observe(prediction) for prediction in predictions]


@router.post("/recommend/")
async def recommend_playlist(
    body: PlaylistRequestSchema,
    cache: REDIS_CONN,
    background_task: BackgroundTasks,
) -> list[PlaylistResponseSchema]:
    """
    Endpoint to get the recommended playlist for users
    """
    predictions, cold_user_no = await get_recommended_playlist(body.user_ids)
    # Updating the Redis Cache
    redis_payload = {k: json.dumps(v) for k, v in predictions.items()}
    await cache.hset(name="playlist", mapping=redis_payload)
    # Update Prometheus from BackgroundTasks
    background_task.add_task(update_prometheus, predictions)

    INFERENCE_COUNT.inc(amount=len(predictions))
    COLD_USER.inc(amount=cold_user_no)

    return [
        PlaylistResponseSchema(user_id=user_id, playlists=playlists)
        for user_id, playlists in predictions.items()
    ]


@router.post("/beta_recommend/")
async def beta_recommend_playlist(
    body: PlaylistRequestSchema,
    # cache: REDIS_CONN,
    # background_task: BackgroundTasks,
) -> dict:
    """
    Beta Endpoint to get the recommended playlist for users
    """
    await send_message_to_queue(json.dumps(body.user_ids))
    return {"status": "send to worker"}
