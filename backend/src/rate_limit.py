"""
Rate Limiting Middleware

Implements rate limiting for API endpoints using slowapi
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from loguru import logger


def get_client_identifier(request: Request) -> str:
    """
    Get client identifier for rate limiting

    Priority:
    1. API key (if authentication is implemented)
    2. IP address
    """
    # Check for API key in header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"

    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=["200/hour", "50/minute"],  # Global limits
    storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
    strategy="fixed-window",
    headers_enabled=True,
)


# Rate limit configurations for different endpoint types
class RateLimits:
    """Rate limit configurations"""

    # Health check - very permissive
    HEALTH = "1000/minute"

    # Single predictions - moderate
    PREDICTION = "100/minute"
    PREDICTION_EXPLAIN = "50/minute"

    # Batch operations - restrictive
    BATCH_PREDICT = "10/hour"

    # Model operations - moderate
    MODEL_INFO = "200/minute"
    MODEL_METRICS = "200/minute"
    FEATURE_IMPORTANCE = "200/minute"

    # Data operations - moderate
    DATA_STATS = "200/minute"

    # Comparison operations - moderate
    COMPARE_MODELS = "50/minute"


# Custom rate limit exceeded handler
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors
    """
    logger.warning(
        f"Rate limit exceeded for {get_client_identifier(request)} "
        f"on {request.url.path}"
    )

    return Response(
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "limit": exc.detail,
        },
        status_code=429,
        headers={
            "Retry-After": str(exc.detail.split()[-1] if exc.detail else "60"),
            "X-RateLimit-Limit": str(exc.detail),
        }
    )


# Decorator for applying rate limits to specific endpoints
def rate_limit(limit: str):
    """
    Apply rate limit to an endpoint

    Usage:
        @rate_limit("10/minute")
        async def my_endpoint():
            ...
    """
    return limiter.limit(limit)
