import asyncio
import json
import aio_pika
from aio_pika import PlainCredentials

from config import settings
from ml_model.model import Recommender
from services.redis import get_async_redis_session


model = Recommender()
model.load_model(settings.MODEL_PATH)


async def worker():
    connection = await aio_pika.connect_robust(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        credentials=PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PWD),
    )

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("inference_queue")

        async for message in queue:
            async with message.process():
                print("Received message")

                cache = await get_async_redis_session()

                user_ids_str = message.body.decode()
                user_ids = json.loads(user_ids_str)  # Convert string back to list

                # Perform prediction with the Recommender model
                predictions = model.predict(user_ids)
                redis_payload = {k: json.dumps(v) for k, v in predictions.items()}
                await cache.hset(name="playlist", mapping=redis_payload)

                # Do some work with the message here (e.g., inference)


# Run the worker
loop = asyncio.get_event_loop()
loop.run_until_complete(worker())
