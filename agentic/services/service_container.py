"""
Service Container

Centralized service management and dependency injection.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Import services
# ChromaDB removed - using MCP server instead
from ..memory.conversation_memory import ConversationMemory
from ..cache.smart_cache import SmartCache
from ..services.llm_client import UnifiedLLMClient
from ..analytics.query_answer_repository import QueryAnswerRepository
from ..analytics.performance_monitor import PerformanceMonitor
from ..analytics.validation_system import ValidationSystem
from ..analytics.analytics_dashboard import AnalyticsDashboard
from ..mcp import MCPServer, MCPSQLiteServer

logger = logging.getLogger(__name__)

# Global service container instance
_service_container_instance: Optional['ServiceContainer'] = None

def get_service_container(config: Dict[str, Any]) -> 'ServiceContainer':
    """Get or create the global service container instance."""
    global _service_container_instance
    if _service_container_instance is None:
        _service_container_instance = ServiceContainer(config)
    return _service_container_instance


class ServiceContainer:
    """
    Centralized service container for dependency injection.
    
    Manages all shared services and their lifecycle.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
        # Debug logging for MCP SQLite config
        mcp_sqlite_config = self.config.get("mcp_sqlite", {})
        logger.info(f"MCP SQLite config: {mcp_sqlite_config}")
        logger.info(f"MCP SQLite enabled: {mcp_sqlite_config.get('enabled', False)}")
        
        # Initialize services synchronously (MCP server will be started lazily)
        self._initialize_services_sync()
        
        logger.info("Service container initialized")
    
    def _initialize_services_sync(self):
        """Initialize services synchronously (MCP server will be started lazily)."""
        # Initialize MCP server without starting it
        mcp_sqlite_config = self.config.get("mcp_sqlite", {})
        
        if mcp_sqlite_config.get("enabled", False):
            logger.info("Using MCP SQLite server (will start lazily)")
            from ..mcp.mcp_sqlite_server import MCPSQLiteServer
            self._services["mcp_server"] = MCPSQLiteServer(mcp_sqlite_config)
        else:
            logger.info("Using legacy MCP server")
            from ..mcp.mcp_server import MCPServer
            mcp_config = self.config.get("mcp", {})
            self._services["mcp_server"] = MCPServer(mcp_config)
        
        # Initialize memory
        memory_config = self.config.get("data", {}).get("memory_config", {})
        from ..memory.conversation_memory import ConversationMemory
        self._services["memory"] = ConversationMemory(memory_config)
        
        # Initialize cache
        cache_config = self.config.get("cache", {})
        from ..cache.smart_cache import SmartCache
        self._services["cache"] = SmartCache(cache_config)
        
        # Initialize LLM client
        llm_config = self.config.get("llm", {})
        from ..services.llm_client import UnifiedLLMClient
        self._services["llm_client"] = UnifiedLLMClient(llm_config)
        
        # Initialize Discord service (requires cache and memory)
        discord_config = self.config.get("discord", {})
        from ..services.discord_service import DiscordMessageService
        # We'll initialize this after cache and memory are available
        # For now, just store the config
        self._discord_config = discord_config
        
        # Initialize sync service
        sync_config = self.config.get("sync", {})
        from ..services.sync_service import DataSynchronizationService
        self._services["sync_service"] = DataSynchronizationService(sync_config)
        
        # Initialize analytics services
        analytics_config = self.config.get("analytics", {})
        from ..analytics.query_answer_repository import QueryAnswerRepository
        from ..analytics.performance_monitor import PerformanceMonitor
        from ..analytics.validation_system import ValidationSystem
        from ..analytics.analytics_dashboard import AnalyticsDashboard
        self._services["query_repository"] = QueryAnswerRepository(analytics_config)
        self._services["performance_monitor"] = PerformanceMonitor(analytics_config)
        self._services["validation_system"] = ValidationSystem(analytics_config)
        self._services["analytics_dashboard"] = AnalyticsDashboard(analytics_config)
        
        # Initialize Discord service with dependencies
        discord_token = self._discord_config.get("token") or os.getenv("DISCORD_TOKEN")
        if discord_token:
            self._services["discord_service"] = DiscordMessageService(
                token=discord_token,
                cache=self._services["cache"],
                memory=self._services["memory"]
            )
        else:
            logger.warning("Discord token not found, Discord service not initialized")
        
        # Set up service dependencies
        self._setup_service_dependencies()
        
        self._initialized = True
        logger.info("Services initialized synchronously")
    
    async def initialize_services(self):
        """Start async services (like MCP server)."""
        if not self._initialized:
            logger.warning("Services not initialized synchronously, initializing now")
            self._initialize_services_sync()
        
        try:
            # Start the MCP server if it's enabled and not already started
            mcp_server = self._services.get("mcp_server")
            if mcp_server and hasattr(mcp_server, 'start') and not getattr(mcp_server, '_started', False):
                logger.info("Starting MCP server...")
                await mcp_server.start()
                logger.info("MCP server started successfully")
            
            logger.info("Async services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to start async services: {e}")
            raise
    
    def _setup_service_dependencies(self):
        """Set up dependencies between services."""
        try:
            # Set up MCP server dependencies
            if "mcp_server" in self._services:
                # MCP server can use LLM client for query translation
                if "llm_client" in self._services:
                    # The MCP server already has its own LLM client, but we can share configuration
                    pass
            
            # Set up analytics dependencies
            if "analytics_dashboard" in self._services:
                if "performance_monitor" in self._services:
                    # Analytics dashboard can use performance monitor
                    pass
            
            logger.info("Service dependencies configured")
            
        except Exception as e:
            logger.error(f"Failed to set up service dependencies: {e}")
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service by name."""
        if not self._initialized:
            logger.warning("Services not initialized, call initialize_services() first")
            return None
        
        return self._services.get(service_name)
    
    def get_mcp_server(self) -> Optional[MCPServer]:
        """Get the MCP server instance."""
        return self.get_service("mcp_server")
    
    def get_memory(self) -> Optional[ConversationMemory]:
        """Get the memory service instance."""
        return self.get_service("memory")
    
    def get_cache(self) -> Optional[SmartCache]:
        """Get the cache service instance."""
        return self.get_service("cache")
    
    def get_llm_client(self) -> Optional[UnifiedLLMClient]:
        """Get the LLM client instance."""
        return self.get_service("llm_client")
    
    def get_query_repository(self) -> Optional[QueryAnswerRepository]:
        """Get the query repository instance."""
        return self.get_service("query_repository")
    
    def get_performance_monitor(self) -> Optional[PerformanceMonitor]:
        """Get the performance monitor instance."""
        return self.get_service("performance_monitor")
    
    def get_validation_system(self) -> Optional[ValidationSystem]:
        """Get the validation system instance."""
        return self.get_service("validation_system")
    
    def get_analytics_dashboard(self) -> Optional[AnalyticsDashboard]:
        """Get the analytics dashboard instance."""
        return self.get_service("analytics_dashboard")
    
    def inject_services(self, target_object: Any):
        """Inject services into a target object that has service attributes."""
        try:
            if hasattr(target_object, 'mcp_server'):
                target_object.mcp_server = self.get_mcp_server()
            if hasattr(target_object, 'memory'):
                target_object.memory = self.get_memory()
            if hasattr(target_object, 'cache'):
                target_object.cache = self.get_cache()
            if hasattr(target_object, 'llm_client'):
                target_object.llm_client = self.get_llm_client()
            if hasattr(target_object, 'query_repository'):
                target_object.query_repository = self.get_query_repository()
            if hasattr(target_object, 'performance_monitor'):
                target_object.performance_monitor = self.get_performance_monitor()
            
            logger.debug(f"Injected services into {type(target_object).__name__}")
            
        except Exception as e:
            logger.error(f"Error injecting services: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": None
        }
        
        try:
            from datetime import datetime
            health_status["timestamp"] = datetime.utcnow().isoformat()
            
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'health_check'):
                        health = await service.health_check()
                        health_status["services"][service_name] = health
                        
                        # Check if any service is unhealthy
                        if health.get("status") != "healthy":
                            health_status["status"] = "degraded"
                    else:
                        health_status["services"][service_name] = {
                            "status": "unknown",
                            "message": "No health check method available"
                        }
                        
                except Exception as e:
                    health_status["services"][service_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "unhealthy"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def close(self):
        """Close all services and cleanup resources."""
        try:
            logger.info("Closing all services...")
            
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'close'):
                        await service.close()
                        logger.debug(f"Closed service: {service_name}")
                    elif hasattr(service, 'stop'):
                        await service.stop()
                        logger.debug(f"Stopped service: {service_name}")
                except Exception as e:
                    logger.warning(f"Error closing service {service_name}: {e}")
            
            self._services.clear()
            self._initialized = False
            
            logger.info("All services closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing services: {e}")
            raise 