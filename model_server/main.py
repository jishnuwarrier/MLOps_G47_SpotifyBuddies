# Built-in
import asyncio

# Third-Party
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


# Local Paths
from config import settings
from routers import routers
from lifespan import lifespan


app = FastAPI(
    title="ML Model Inference Service",
    description="This service to expose the API endpoint to interact with the model",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(routers)


# Expose the app to prometheus
if not settings.DEBUG:
    print("Connecting with prometheus")
    Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running or not
    """
    await asyncio.sleep(1)
    print(settings)
    return {"Status": "Server Running"}
