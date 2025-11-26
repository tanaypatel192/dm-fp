"""
Application Configuration

Manages all configuration from environment variables
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Application
    app_name: str = "Diabetes Prediction API"
    environment: str = "development"  # development, staging, production
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite+aiosqlite:///./diabetes_predictions.db"

    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_prediction: int = 3600
    cache_ttl_model_metrics: int = 7200

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_storage: str = "memory://"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    log_json_format: bool = False

    # Security
    secret_key: str = "change-this-in-production"
    api_key_required: bool = False
    api_keys: str = ""  # Comma-separated
    cors_origins: str = "http://localhost:3000"

    # HTTPS
    https_only: bool = False
    secure_cookies: bool = False

    # Model Configuration
    default_model: str = "xgboost"
    model_path: str = "models/"
    enable_shap: bool = True
    max_batch_size: int = 1000

    # Performance
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_connections: int = 1000
    keepalive: int = 5
    timeout: int = 120

    # Monitoring
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_api_keys_list(self) -> List[str]:
        """Get list of API keys"""
        if not self.api_keys:
            return []
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    def get_cors_origins_list(self) -> List[str]:
        """Get list of CORS origins"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
