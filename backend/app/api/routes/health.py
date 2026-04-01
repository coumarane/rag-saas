from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Return service liveness status, API version, and current environment."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
    )
