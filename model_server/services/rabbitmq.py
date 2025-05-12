import aio_pika

from config import settings


async def get_rabbitmq_connection():
    return await aio_pika.connect_robust(
        f"ampq://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PWD}@{settings.RABBITMQ_HOST}",
    )


async def send_message_to_queue(message: str):
    connection = await get_rabbitmq_connection()
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("inference_queue")
        await channel.default_exchange.publish(
            aio_pika.Message(body=message.encode()),
            routing_key=queue.name,
        )
