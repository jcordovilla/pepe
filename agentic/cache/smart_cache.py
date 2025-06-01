"""
Smart Multi-Level Cache System

Provides memory, Redis, and file-based caching with intelligent cache management.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import asyncio
import aiofiles

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from cachetools import TTLCache, LRUCache

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Multi-level cache with memory, Redis, and file persistence.
    
    Features:
    - Memory cache (L1) - Fastest access
    - Redis cache (L2) - Shared across instances
    - File cache (L3) - Persistent across restarts
    - Intelligent cache promotion/demotion
    - Automatic cleanup and expiration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Memory cache configuration
        self.memory_maxsize = config.get("memory_maxsize", 1000)
        self.memory_ttl = config.get("memory_ttl", 300)  # 5 minutes
        
        # Redis configuration
        self.redis_enabled = config.get("redis_enabled", False) and REDIS_AVAILABLE
        self.redis_url = config.get("redis_url", "redis://localhost:6379")
        self.redis_ttl = config.get("redis_ttl", 3600)  # 1 hour
        self.redis_prefix = config.get("redis_prefix", "rag_cache:")
        
        # File cache configuration
        self.file_cache_enabled = config.get("file_cache_enabled", True)
        self.file_cache_dir = config.get("file_cache_dir", "./data/cache")
        self.file_cache_ttl = config.get("file_cache_ttl", 86400)  # 24 hours
        self.file_cache_maxsize = config.get("file_cache_maxsize", 10000)
        
        # Initialize caches
        self._init_memory_cache()
        self._init_redis_cache()
        self._init_file_cache()
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "file_hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
        
        logger.info("SmartCache initialized successfully")
    
    def _init_memory_cache(self):
        """Initialize memory cache."""
        self.memory_cache = TTLCache(
            maxsize=self.memory_maxsize,
            ttl=self.memory_ttl
        )
        logger.info(f"Memory cache initialized: {self.memory_maxsize} items, {self.memory_ttl}s TTL")
    
    def _init_redis_cache(self):
        """Initialize Redis cache."""
        self.redis_client = None
        
        if self.redis_enabled:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False  # We'll handle binary data
                )
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_enabled = False
    
    def _init_file_cache(self):
        """Initialize file cache."""
        if self.file_cache_enabled:
            os.makedirs(self.file_cache_dir, exist_ok=True)
            logger.info(f"File cache initialized: {self.file_cache_dir}")
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking all levels.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            # Level 1: Memory cache
            if key in self.memory_cache:
                self.stats["memory_hits"] += 1
                return self.memory_cache[key]
            
            # Level 2: Redis cache
            if self.redis_enabled and self.redis_client:
                try:
                    redis_key = f"{self.redis_prefix}{self._hash_key(key)}"
                    redis_value = await self.redis_client.get(redis_key)
                    
                    if redis_value:
                        value = pickle.loads(redis_value)
                        
                        # Promote to memory cache
                        self.memory_cache[key] = value
                        
                        self.stats["redis_hits"] += 1
                        return value
                        
                except Exception as e:
                    logger.warning(f"Redis cache error: {e}")
            
            # Level 3: File cache
            if self.file_cache_enabled:
                try:
                    file_path = self._get_file_cache_path(key)
                    
                    if os.path.exists(file_path):
                        async with aiofiles.open(file_path, 'rb') as f:
                            data = await f.read()
                            cache_data = pickle.loads(data)
                        
                        # Check expiration
                        if cache_data.get("expires_at", 0) > datetime.utcnow().timestamp():
                            value = cache_data["value"]
                            
                            # Promote to higher levels
                            self.memory_cache[key] = value
                            
                            if self.redis_enabled and self.redis_client:
                                redis_key = f"{self.redis_prefix}{self._hash_key(key)}"
                                await self.redis_client.setex(
                                    redis_key,
                                    self.redis_ttl,
                                    pickle.dumps(value)
                                )
                            
                            self.stats["file_hits"] += 1
                            return value
                        else:
                            # Expired, remove file
                            os.remove(file_path)
                            
                except Exception as e:
                    logger.warning(f"File cache error: {e}")
            
            # Not found in any cache
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache at all levels.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful
        """
        try:
            # Level 1: Memory cache
            self.memory_cache[key] = value
            
            # Level 2: Redis cache
            if self.redis_enabled and self.redis_client:
                try:
                    redis_key = f"{self.redis_prefix}{self._hash_key(key)}"
                    redis_ttl = ttl or self.redis_ttl
                    
                    await self.redis_client.setex(
                        redis_key,
                        redis_ttl,
                        pickle.dumps(value)
                    )
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            # Level 3: File cache
            if self.file_cache_enabled:
                try:
                    file_path = self._get_file_cache_path(key)
                    cache_data = {
                        "value": value,
                        "created_at": datetime.utcnow().timestamp(),
                        "expires_at": (datetime.utcnow() + timedelta(seconds=ttl or self.file_cache_ttl)).timestamp()
                    }
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(pickle.dumps(cache_data))
                        
                except Exception as e:
                    logger.warning(f"File cache set error: {e}")
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from all cache levels.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful
        """
        try:
            success = True
            
            # Level 1: Memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Level 2: Redis cache
            if self.redis_enabled and self.redis_client:
                try:
                    redis_key = f"{self.redis_prefix}{self._hash_key(key)}"
                    await self.redis_client.delete(redis_key)
                except Exception as e:
                    logger.warning(f"Redis delete error: {e}")
                    success = False
            
            # Level 3: File cache
            if self.file_cache_enabled:
                try:
                    file_path = self._get_file_cache_path(key)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"File cache delete error: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all caches."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_enabled and self.redis_client:
                try:
                    pattern = f"{self.redis_prefix}*"
                    keys = []
                    async for key in self.redis_client.scan_iter(match=pattern):
                        keys.append(key)
                    
                    if keys:
                        await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")
            
            # Clear file cache
            if self.file_cache_enabled and os.path.exists(self.file_cache_dir):
                try:
                    import shutil
                    shutil.rmtree(self.file_cache_dir)
                    os.makedirs(self.file_cache_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"File cache clear error: {e}")
            
            logger.info("All caches cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            return False
    
    def _get_file_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        hash_key = self._hash_key(key)
        # Create subdirectories based on first two characters
        subdir = hash_key[:2]
        return os.path.join(self.file_cache_dir, subdir, f"{hash_key}.cache")
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        
        try:
            # File cache cleanup
            if self.file_cache_enabled and os.path.exists(self.file_cache_dir):
                current_time = datetime.utcnow().timestamp()
                
                for root, dirs, files in os.walk(self.file_cache_dir):
                    for file in files:
                        if file.endswith('.cache'):
                            file_path = os.path.join(root, file)
                            try:
                                async with aiofiles.open(file_path, 'rb') as f:
                                    data = await f.read()
                                    cache_data = pickle.loads(data)
                                
                                if cache_data.get("expires_at", 0) <= current_time:
                                    os.remove(file_path)
                                    cleaned += 1
                                    
                            except Exception as e:
                                # If we can't read the file, remove it
                                try:
                                    os.remove(file_path)
                                    cleaned += 1
                                except:
                                    pass
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired cache entries")
                
            return cleaned
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum([
            self.stats["memory_hits"],
            self.stats["redis_hits"],
            self.stats["file_hits"],
            self.stats["misses"]
        ])
        
        hit_rate = 0
        if total_requests > 0:
            hits = total_requests - self.stats["misses"]
            hit_rate = (hits / total_requests) * 100
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_maxsize": self.memory_maxsize
        }
    
    async def close(self):
        """Close cache connections."""
        if self.redis_enabled and self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
