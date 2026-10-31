from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Endpoint to verify backend process health, version, and status."
)
def get_health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mock_mode": settings.use_mock_inference,
    }
