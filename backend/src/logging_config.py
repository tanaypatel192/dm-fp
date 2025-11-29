"""
Logging Configuration with Loguru

Structured logging for all application events
"""

import sys
import json
from pathlib import Path
from loguru import logger
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    json_format: bool = False
):
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_dir: Directory for log files
        json_format: Whether to use JSON format for logs
    """
    # Remove default logger
    logger.remove()

    # Console logging
    if json_format:
        # JSON format for production
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Human-readable format for development
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File logging
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # General log file
        logger.add(
            log_path / "app.log",
            level=log_level,
            rotation="500 MB",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )

        # Error log file
        logger.add(
            log_path / "error.log",
            level="ERROR",
            rotation="500 MB",
            retention="60 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )

        # API request log file (JSON format)
        logger.add(
            log_path / "api_requests.log",
            level="INFO",
            rotation="500 MB",
            retention="30 days",
            compression="zip",
            serialize=True,
            filter=lambda record: "api_request" in record["extra"],
        )

        # Performance metrics log file
        logger.add(
            log_path / "performance.log",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            serialize=True,
            filter=lambda record: "performance" in record["extra"],
        )

    logger.info(f"Logging configured - Level: {log_level}, File logging: {log_to_file}")


# Logging helpers
class LoggingHelper:
    """Helper functions for structured logging"""

    @staticmethod
    def log_api_request(
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        ip_address: str = None,
        user_agent: str = None,
    ):
        """Log API request with structured data"""
        logger.bind(api_request=True).info(
            "API Request",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    @staticmethod
    def log_prediction(
        model_name: str,
        prediction: int,
        probability: float,
        confidence: float,
        duration_ms: float,
    ):
        """Log prediction with structured data"""
        logger.bind(performance=True).info(
            "Prediction",
            extra={
                "model_name": model_name,
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    @staticmethod
    def log_batch_job(
        job_id: str,
        status: str,
        total_patients: int,
        processed: int,
        duration_ms: float = None,
    ):
        """Log batch job with structured data"""
        logger.bind(performance=True).info(
            "Batch Job",
            extra={
                "job_id": job_id,
                "status": status,
                "total_patients": total_patients,
                "processed": processed,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    @staticmethod
    def log_cache_operation(
        operation: str,
        key: str,
        hit: bool = None,
        duration_ms: float = None,
    ):
        """Log cache operation"""
        logger.bind(performance=True).debug(
            "Cache Operation",
            extra={
                "operation": operation,
                "key": key,
                "hit": hit,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        context: dict = None,
        exc_info: Exception = None,
    ):
        """Log error with context"""
        extra_data = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if context:
            extra_data.update(context)

        if exc_info:
            logger.exception(f"Error: {error_type}", extra=extra_data)
        else:
            logger.error(f"Error: {error_type}", extra=extra_data)


# Performance monitoring context manager
class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, log_result: bool = True):
        self.operation_name = operation_name
        self.log_result = log_result
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        from time import perf_counter
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from time import perf_counter
        self.duration_ms = (perf_counter() - self.start_time) * 1000

        if self.log_result:
            if exc_type:
                logger.bind(performance=True).warning(
                    f"{self.operation_name} failed",
                    duration_ms=self.duration_ms,
                    error=str(exc_val) if exc_val else None,
                )
            else:
                logger.bind(performance=True).info(
                    f"{self.operation_name}",
                    duration_ms=self.duration_ms,
                )


# Request ID middleware for tracing
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    """Get current request ID"""
    return request_id_var.get()


def set_request_id(request_id: str = None):
    """Set request ID for current context"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id
