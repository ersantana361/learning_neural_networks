from fastapi import APIRouter
import docker
from docker.errors import DockerException

from ..config import settings

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health_check():
    return {"status": "healthy", "service": "nn-backend"}


@router.get("/executor")
async def executor_health():
    try:
        client = docker.from_env()
        client.ping()

        # Check if executor image exists
        try:
            client.images.get(settings.executor_image)
            image_status = "available"
        except docker.errors.ImageNotFound:
            image_status = "not_found"

        return {
            "status": "healthy",
            "docker": "connected",
            "executor_image": settings.executor_image,
            "image_status": image_status,
        }
    except DockerException as e:
        return {
            "status": "unhealthy",
            "docker": "disconnected",
            "error": str(e),
        }
