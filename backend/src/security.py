"""
Security Middleware and Utilities

Input validation, sanitization, and security headers
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Callable
import re
from loguru import logger


# API Key authentication (optional)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for adding security headers"""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove sensitive headers
        response.headers.pop("Server", None)

        return response


class InputValidator:
    """Input validation and sanitization"""

    @staticmethod
    def validate_patient_data(data: dict) -> dict:
        """
        Validate and sanitize patient data

        Ensures all required fields are present and within valid ranges
        """
        required_fields = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]

        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # Validate ranges
        validations = {
            "Pregnancies": (0, 20, "integer"),
            "Glucose": (0, 300, "float"),
            "BloodPressure": (0, 200, "float"),
            "SkinThickness": (0, 100, "float"),
            "Insulin": (0, 900, "float"),
            "BMI": (0, 100, "float"),
            "DiabetesPedigreeFunction": (0, 3, "float"),
            "Age": (0, 120, "integer"),
        }

        sanitized_data = {}
        for field, (min_val, max_val, val_type) in validations.items():
            value = data.get(field)

            # Type conversion
            try:
                if val_type == "integer":
                    value = int(value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid value for {field}: must be a {val_type}"
                )

            # Range validation
            if not (min_val <= value <= max_val):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid value for {field}: must be between {min_val} and {max_val}"
                )

            sanitized_data[field] = value

        return sanitized_data

    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """Validate model name"""
        valid_models = ["decision_tree", "random_forest", "xgboost"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name. Must be one of: {', '.join(valid_models)}"
            )
        return model_name

    @staticmethod
    def validate_top_n(top_n: int) -> int:
        """Validate top_n parameter"""
        if not (1 <= top_n <= 50):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_n must be between 1 and 50"
            )
        return top_n

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """
        Sanitize string input

        Removes potentially harmful characters and limits length
        """
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)

        # Remove SQL injection attempts
        sanitized = re.sub(r'(\'|\"|\;|\-\-|\/\*|\*\/)', '', sanitized)

        # Limit length
        sanitized = sanitized[:max_length]

        return sanitized.strip()

    @staticmethod
    def validate_batch_size(size: int) -> int:
        """Validate batch size"""
        max_batch_size = 1000
        if not (1 <= size <= max_batch_size):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size must be between 1 and {max_batch_size}"
            )
        return size


class APIKeyValidator:
    """API key validation (optional authentication)"""

    def __init__(self, api_keys: Optional[set] = None):
        self.api_keys = api_keys or set()
        self.enabled = len(self.api_keys) > 0

    async def validate_api_key(self, api_key: Optional[str] = None) -> bool:
        """Validate API key"""
        if not self.enabled:
            return True  # API key validation disabled

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return True


# SQL Injection prevention helpers
def escape_sql_string(value: str) -> str:
    """Escape SQL special characters"""
    return value.replace("'", "''").replace(";", "").replace("--", "")


# XSS prevention helpers
def escape_html(value: str) -> str:
    """Escape HTML special characters"""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# Request ID tracking for security auditing
async def get_client_info(request: Request) -> dict:
    """Extract client information from request"""
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "referer": request.headers.get("referer"),
        "origin": request.headers.get("origin"),
    }


# Content Security Policy
def get_csp_header() -> str:
    """Get Content Security Policy header"""
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )


# CORS configuration
def get_cors_config(environment: str = "production") -> dict:
    """Get CORS configuration based on environment"""
    if environment == "development":
        return {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    else:
        # Production: restrict to specific origins
        return {
            "allow_origins": [
                "https://yourdomain.com",
                "https://www.yourdomain.com",
            ],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
        }
