# Built-in
import asyncio

# Third-Party
from fastapi import FastAPI

# Prometheus
from prometheus_fastapi_instrumentator import Instrumentator

# Local Paths
from config import settings
from routers import routers


app = FastAPI(
    title="ML Model Inference Service",
    description="This service to expose the API endpoint to interact with the model",
    version="1.0.0",
)

app.include_router(routers)


# Expose the App to prometheus
if not settings.DEBUG:
    Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running or not
    """
    await asyncio.sleep(1)
    print(settings)
    return {"Status": "Server Running"}
