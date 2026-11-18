from fastapi import APIRouter
from app.api.v1.endpoints import files, health, models, probe

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(models.router, tags=["Models"])
api_router.include_router(probe.router, tags=["Model Probing"])
api_router.include_router(files.router, tags=["File Processing"])
