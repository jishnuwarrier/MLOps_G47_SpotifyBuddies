from fastapi import APIRouter

from . import playlist

prefix = r"/api"
routers = APIRouter(prefix=prefix)
routers.include_router(playlist.router)
