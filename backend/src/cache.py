"""
Caching Utility with Redis

Provides caching functionality for predictions, model metrics, and data
"""

import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
import redis.asyncio as redis
from loguru import logger


class CacheManager:
    """Redis cache manager for predictions and metrics"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.enabled = True

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("✓ Redis connection established")
            self.enabled = True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.enabled = False
            self.redis = None

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache with optional TTL (seconds)"""
        if not self.enabled or not self.redis:
            return

        try:
            serialized = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def delete(self, key: str):
        """Delete value from cache"""
        if not self.enabled or not self.redis:
            return

        try:
            await self.redis.delete(key)
            logger.debug(f"Cache DELETE: {key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    async def delete_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        if not self.enabled or not self.redis:
            return

        try:
            keys = []
            async for key in self.redis.scan_iter(pattern):
                keys.append(key)

            if keys:
                await self.redis.delete(*keys)
                logger.debug(f"Cache DELETE pattern: {pattern} ({len(keys)} keys)")
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")

    async def clear(self):
        """Clear all cache"""
        if not self.enabled or not self.redis:
            return

        try:
            await self.redis.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a string representation of args and kwargs
        key_data = str(args) + str(sorted(kwargs.items()))
        # Hash it for consistent key length
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return key_hash


# Global cache instance
cache = CacheManager()


# Cache TTL constants (seconds)
class CacheTTL:
    PREDICTION = 3600  # 1 hour
    MODEL_METRICS = 7200  # 2 hours
    FEATURE_IMPORTANCE = 7200  # 2 hours
    DATA_STATS = 3600  # 1 hour
    HEALTH_CHECK = 60  # 1 minute
    BATCH_RESULT = 1800  # 30 minutes


def cached(
    ttl: int = 3600,
    key_prefix: str = "cache",
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_func: Optional function to generate cache key from args
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{key_prefix}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{key_prefix}:{CacheManager.generate_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# Cache key generators for common patterns
class CacheKeys:
    """Cache key generators"""

    @staticmethod
    def prediction(patient_data: dict, model_name: str) -> str:
        """Generate key for prediction cache"""
        data_hash = CacheManager.generate_key(patient_data)
        return f"prediction:{model_name}:{data_hash}"

    @staticmethod
    def model_metrics(model_name: str) -> str:
        """Generate key for model metrics cache"""
        return f"model_metrics:{model_name}"

    @staticmethod
    def feature_importance(model_name: str, top_n: int) -> str:
        """Generate key for feature importance cache"""
        return f"feature_importance:{model_name}:{top_n}"

    @staticmethod
    def data_stats() -> str:
        """Generate key for data stats cache"""
        return "data_stats:current"

    @staticmethod
    def batch_result(batch_id: str) -> str:
        """Generate key for batch result cache"""
        return f"batch_result:{batch_id}"


# Invalidation helpers
async def invalidate_model_cache(model_name: str):
    """Invalidate all cache entries for a model"""
    patterns = [
        f"prediction:{model_name}:*",
        f"model_metrics:{model_name}",
        f"feature_importance:{model_name}:*",
    ]
    for pattern in patterns:
        await cache.delete_pattern(pattern)
    logger.info(f"Invalidated cache for model: {model_name}")


async def invalidate_all_predictions():
    """Invalidate all prediction cache"""
    await cache.delete_pattern("prediction:*")
    logger.info("Invalidated all prediction cache")


# Cache warming functions
async def warm_model_cache(model_name: str, model_data: dict):
    """Pre-warm cache with model metrics"""
    key = CacheKeys.model_metrics(model_name)
    await cache.set(key, model_data, CacheTTL.MODEL_METRICS)
    logger.info(f"Warmed cache for model: {model_name}")


async def warm_feature_importance_cache(model_name: str, top_n: int, data: dict):
    """Pre-warm cache with feature importance"""
    key = CacheKeys.feature_importance(model_name, top_n)
    await cache.set(key, data, CacheTTL.FEATURE_IMPORTANCE)
    logger.info(f"Warmed feature importance cache for model: {model_name}")
