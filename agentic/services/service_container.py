"""
Service Container for Dependency Injection

Provides shared services to all agents to eliminate resource duplication.
Fixes the issue where each agent initializes its own LLM client and other services.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..services.llm_client import UnifiedLLMClient
from ..vectorstore.persistent_store import PersistentVectorStore
from ..cache.smart_cache import SmartCache
from ..memory.conversation_memory import ConversationMemory
from ..analytics.query_answer_repository import QueryAnswerRepository
from ..analytics.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """
    Container for shared services used across all agents.
    
    Provides:
    - Unified LLM client with connection pooling
    - Shared vector store instance
    - Centralized caching
    - Memory management
    - Analytics services
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the service container with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize shared services
        self._initialize_services()
        
        logger.info("ServiceContainer initialized with all shared services")
    
    def _initialize_services(self):
        """Initialize all shared services."""
        try:
            # LLM Client (shared across all agents)
            llm_config = self.config.get("llm", {})
            self.llm_client = UnifiedLLMClient(llm_config)
            
            # Vector Store (shared across search and analysis agents)
            vectorstore_config = self.config.get("vectorstore", {})
            self.vector_store = PersistentVectorStore(vectorstore_config)
            
            # Cache (shared across all agents)
            cache_config = self.config.get("cache", {})
            self.cache = SmartCache(cache_config)
            
            # Memory (shared across all agents)
            memory_config = self.config.get("memory", {})
            self.memory = ConversationMemory(memory_config)
            
            # Analytics (shared across all agents)
            analytics_config = self.config.get("analytics", {})
            self.query_repository = QueryAnswerRepository(analytics_config)
            self.performance_monitor = PerformanceMonitor(analytics_config)
            
            # Set up service dependencies
            self._setup_service_dependencies()
            
            logger.info("All shared services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing service container: {e}")
            raise
    
    def _setup_service_dependencies(self):
        """Set up dependencies between services."""
        try:
            # Set up performance monitor with its dependencies
            self.performance_monitor.set_components(
                self.query_repository,
                self.vector_store,
                self.cache
            )
            
            logger.info("Service dependencies configured successfully")
            
        except Exception as e:
            logger.error(f"Error setting up service dependencies: {e}")
            raise
    
    def inject_services(self, agent_instance: Any) -> None:
        """
        Inject shared services into an agent instance.
        
        Args:
            agent_instance: The agent instance to inject services into
        """
        try:
            # Inject LLM client
            if hasattr(agent_instance, 'llm_client'):
                agent_instance.llm_client = self.llm_client
            
            # Inject vector store
            if hasattr(agent_instance, 'vector_store'):
                agent_instance.vector_store = self.vector_store
            
            # Inject cache
            if hasattr(agent_instance, 'cache'):
                agent_instance.cache = self.cache
            
            # Inject memory
            if hasattr(agent_instance, 'memory'):
                agent_instance.memory = self.memory
            
            # Inject analytics services
            if hasattr(agent_instance, 'query_repository'):
                agent_instance.query_repository = self.query_repository
            
            if hasattr(agent_instance, 'performance_monitor'):
                agent_instance.performance_monitor = self.performance_monitor
            
            logger.debug(f"Services injected into {type(agent_instance).__name__}")
            
        except Exception as e:
            logger.error(f"Error injecting services into {type(agent_instance).__name__}: {e}")
            raise
    
    def get_llm_client(self) -> UnifiedLLMClient:
        """Get the shared LLM client."""
        return self.llm_client
    
    def get_vector_store(self) -> PersistentVectorStore:
        """Get the shared vector store."""
        return self.vector_store
    
    def get_cache(self) -> SmartCache:
        """Get the shared cache."""
        return self.cache
    
    def get_memory(self) -> ConversationMemory:
        """Get the shared memory."""
        return self.memory
    
    def get_query_repository(self) -> QueryAnswerRepository:
        """Get the shared query repository."""
        return self.query_repository
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get the shared performance monitor."""
        return self.performance_monitor
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all services.
        
        Returns:
            Health status of all services
        """
        health_status = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": None
        }
        
        try:
            # Check LLM client
            llm_health = await self.llm_client.health_check()
            health_status["services"]["llm_client"] = llm_health
            
            # Check vector store
            try:
                # Simple health check for vector store
                collections = await self.vector_store.list_collections()
                health_status["services"]["vector_store"] = {
                    "status": "healthy",
                    "collections_count": len(collections)
                }
            except Exception as e:
                health_status["services"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            # Check cache
            try:
                # Simple health check for cache
                await self.cache.set("health_check", "ok", ttl=60)
                test_value = await self.cache.get("health_check")
                health_status["services"]["cache"] = {
                    "status": "healthy" if test_value == "ok" else "unhealthy"
                }
            except Exception as e:
                health_status["services"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            # Check memory
            try:
                # Simple health check for memory
                await self.memory.get_history("test_user", limit=1)
                health_status["services"]["memory"] = {"status": "healthy"}
            except Exception as e:
                health_status["services"]["memory"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            logger.info(f"Health check completed: {health_status['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        try:
            logger.info("Shutting down service container...")
            
            # Shutdown performance monitor
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.stop_monitoring()
            
            # Close vector store connections
            if hasattr(self, 'vector_store'):
                await self.vector_store.close()
            
            # Clear cache
            if hasattr(self, 'cache'):
                await self.cache.clear()
            
            logger.info("Service container shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during service container shutdown: {e}")


# Global service container instance
_service_container: Optional[ServiceContainer] = None


def get_service_container(config: Optional[Dict[str, Any]] = None) -> ServiceContainer:
    """
    Get the global service container instance.
    
    Args:
        config: Configuration to initialize the container (only used on first call)
        
    Returns:
        Service container instance
    """
    global _service_container
    
    if _service_container is None:
        if config is None:
            raise ValueError("Service container not initialized. Provide config on first call.")
        _service_container = ServiceContainer(config)
    
    return _service_container


def reset_service_container():
    """Reset the global service container (for testing)."""
    global _service_container
    _service_container = None 