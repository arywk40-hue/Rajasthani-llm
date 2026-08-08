"""
Health Check Endpoint
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Returns service health status."""
    return {
        "status": "healthy",
        "service": "rajasthani-dialect-ai",
        "version": "0.1.0",
    }
