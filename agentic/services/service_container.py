"""
Service Container

Provides dependency injection and shared service management for the agentic system.
Reduces resource duplication and improves performance through service sharing.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..vectorstore.persistent_store import PersistentVectorStore
from ..memory.conversation_memory import ConversationMemory
from ..cache.smart_cache import SmartCache
from ..services.llm_client import UnifiedLLMClient
from ..analytics.query_answer_repository import QueryAnswerRepository
from ..analytics.performance_monitor import PerformanceMonitor
from ..analytics.validation_system import ValidationSystem
from ..analytics.analytics_dashboard import AnalyticsDashboard

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Service container for dependency injection and shared service management.
    
    Provides:
    - Shared service instances across all agents
    - Resource optimization and reuse
    - Centralized service configuration
    - Service lifecycle management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
        logger.info("ServiceContainer initialized")
    
    async def initialize_services(self):
        """Initialize all shared services."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing shared services...")
            
            # Initialize vector store
            vector_config = self.config.get("data", {}).get("vector_config", {})
            self._services["vector_store"] = PersistentVectorStore(vector_config)
            
            # Initialize memory
            memory_config = self.config.get("data", {}).get("memory_config", {})
            self._services["memory"] = ConversationMemory(memory_config)
            
            # Initialize cache
            cache_config = self.config.get("data", {}).get("cache_config", {})
            self._services["cache"] = SmartCache(cache_config)
            
            # Initialize LLM client
            llm_config = self.config.get("llm", {})
            self._services["llm_client"] = UnifiedLLMClient(llm_config)
            
            # Initialize analytics services
            analytics_config = self.config.get("analytics", {})
            self._services["query_repository"] = QueryAnswerRepository(analytics_config)
            self._services["performance_monitor"] = PerformanceMonitor(analytics_config)
            self._services["validation_system"] = ValidationSystem(analytics_config)
            self._services["analytics_dashboard"] = AnalyticsDashboard(analytics_config)
            
            # Set up service dependencies
            self._setup_service_dependencies()
            
            self._initialized = True
            logger.info("All shared services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def _setup_service_dependencies(self):
        """Set up dependencies between services."""
        try:
            # Set performance monitor dependencies
            performance_monitor = self._services["performance_monitor"]
            performance_monitor.set_components(
                self._services["query_repository"],
                self._services["vector_store"],
                self._services["cache"]
            )
            
            # Set analytics dashboard dependencies
            analytics_dashboard = self._services["analytics_dashboard"]
            analytics_dashboard.set_components(
                self._services["query_repository"],
                self._services["performance_monitor"],
                self._services["validation_system"]
            )
            
            logger.info("Service dependencies configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup service dependencies: {e}")
            raise
    
    def get_vector_store(self) -> PersistentVectorStore:
        """Get shared vector store instance."""
        return self._services["vector_store"]
    
    def get_memory(self) -> ConversationMemory:
        """Get shared memory instance."""
        return self._services["memory"]
    
    def get_cache(self) -> SmartCache:
        """Get shared cache instance."""
        return self._services["cache"]
    
    def get_llm_client(self) -> UnifiedLLMClient:
        """Get shared LLM client instance."""
        return self._services["llm_client"]
    
    def get_query_repository(self) -> QueryAnswerRepository:
        """Get shared query repository instance."""
        return self._services["query_repository"]
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get shared performance monitor instance."""
        return self._services["performance_monitor"]
    
    def get_validation_system(self) -> ValidationSystem:
        """Get shared validation system instance."""
        return self._services["validation_system"]
    
    def get_analytics_dashboard(self) -> AnalyticsDashboard:
        """Get shared analytics dashboard instance."""
        return self._services["analytics_dashboard"]
    
    def inject_services(self, agent: Any):
        """Inject shared services into an agent."""
        try:
            if hasattr(agent, 'set_services'):
                agent.set_services(
                    vector_store=self._services["vector_store"],
                    memory=self._services["memory"],
                    cache=self._services["cache"],
                    llm_client=self._services["llm_client"],
                    query_repository=self._services["query_repository"],
                    performance_monitor=self._services["performance_monitor"]
                )
            elif hasattr(agent, 'vector_store'):
                agent.vector_store = self._services["vector_store"]
            if hasattr(agent, 'memory'):
                agent.memory = self._services["memory"]
            if hasattr(agent, 'cache'):
                agent.cache = self._services["cache"]
            if hasattr(agent, 'llm_client'):
                agent.llm_client = self._services["llm_client"]
                
            logger.debug(f"Services injected into {type(agent).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to inject services into {type(agent).__name__}: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics about service usage."""
        stats = {
            "initialized": self._initialized,
            "service_count": len(self._services),
            "services": list(self._services.keys())
        }
        
        # Add service-specific stats
        if "cache" in self._services:
            cache_stats = self._services["cache"].get_stats()
            stats["cache"] = cache_stats
        
        if "performance_monitor" in self._services:
            perf_stats = self._services["performance_monitor"].get_stats()
            stats["performance"] = perf_stats
        
        return stats
    
    async def cleanup(self):
        """Clean up all services and resources."""
        try:
            logger.info("Cleaning up service container...")
            
            # Clean up services in reverse order
            cleanup_order = [
                "analytics_dashboard",
                "validation_system", 
                "performance_monitor",
                "query_repository",
                "llm_client",
                "cache",
                "memory",
                "vector_store"
            ]
            
            for service_name in cleanup_order:
                if service_name in self._services:
                    service = self._services[service_name]
                    if hasattr(service, 'cleanup'):
                        await service.cleanup()
                    elif hasattr(service, 'close'):
                        await service.close()
            
            self._services.clear()
            self._initialized = False
            
            logger.info("Service container cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")


# Global service container instance
_service_container: Optional[ServiceContainer] = None


def get_service_container(config: Optional[Dict[str, Any]] = None) -> ServiceContainer:
    """Get or create the global service container instance."""
    global _service_container
    
    if _service_container is None:
        if config is None:
            from ..config.modernized_config import get_modernized_config
            config = get_modernized_config()
        
        _service_container = ServiceContainer(config)
    
    return _service_container


async def initialize_global_services(config: Optional[Dict[str, Any]] = None):
    """Initialize the global service container."""
    container = get_service_container(config)
    await container.initialize_services()


async def cleanup_global_services():
    """Clean up the global service container."""
    global _service_container
    
    if _service_container is not None:
        await _service_container.cleanup()
        _service_container = None 