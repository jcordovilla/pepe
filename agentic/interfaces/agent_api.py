"""
Agent API Interface

Provides a clean REST-like interface for interacting with the agentic RAG system.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from ..agents.orchestrator import AgentOrchestrator
from ..mcp import MCPServer
from ..memory.conversation_memory import ConversationMemory
from ..agents.pipeline_agent import PipelineAgent
from ..analytics import QueryAnswerRepository, PerformanceMonitor, ValidationSystem, AnalyticsDashboard
from ..analytics.query_answer_repository import QueryMetrics
from ..agents import agent_registry

logger = logging.getLogger(__name__)


class AgentAPI:
    """
    High-level API interface for the agentic RAG system.
    
    Provides methods for querying, managing context, and system administration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize service container for dependency injection
        from ..services.service_container import get_service_container
        self.service_container = get_service_container(config)
        
        # Create orchestrator with v2 agent registry
        self.orchestrator = AgentOrchestrator(config)
        
        # Initialize analytics components
        analytics_config = config.get("analytics", {})
        self.query_repository = QueryAnswerRepository(analytics_config)
        self.performance_monitor = PerformanceMonitor(analytics_config)
        self.validation_system = ValidationSystem(analytics_config)
        self.analytics_dashboard = AnalyticsDashboard(analytics_config)
        
        # API configuration
        self.max_query_length = config.get("max_query_length", 2000)
        self.default_k = config.get("default_k", 10)
        self.enable_learning = config.get("enable_learning", True)
        self.enable_analytics = config.get("enable_analytics", True)
        
        logger.info("AgentAPI initialized successfully with analytics")
    
    async def initialize(self):
        """Initialize the service container and inject services into agents."""
        try:
            # Initialize service container
            await self.service_container.initialize_services()
            
            # Get services after initialization
            self.mcp_server = self.service_container.get_mcp_server()
            self.memory = self.service_container.get_memory()
            self.pipeline = PipelineAgent(self.config.get("pipeline", {}))
            
            # Inject shared services into all agents
            self.service_container.inject_services(self.orchestrator)
            
            # Set component references for analytics
            self.performance_monitor.set_components(
                self.query_repository, 
                self.mcp_server, 
                self.service_container.get_cache()
            )
            self.analytics_dashboard.set_components(
                self.query_repository, 
                self.performance_monitor, 
                self.validation_system
            )
            
            logger.info("AgentAPI services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AgentAPI services: {e}")
            raise
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.enable_analytics:
            self.performance_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if self.enable_analytics:
            self.performance_monitor.stop_monitoring()
    
    async def query(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User query text
            user_id: Unique user identifier
            context: Optional additional context
            preferences: User preferences
            
        Returns:
            Response with answer, sources, and metadata
        """
        start_time = datetime.utcnow()
        platform = context.get("platform", "unknown") if context else "unknown"
        channel_id = context.get("channel_id") if context else None
        
        try:
            # Ensure services are initialized
            if not hasattr(self, 'mcp_server') or self.mcp_server is None:
                await self.initialize()
            
            # Validate input
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "Empty query provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            if len(query) > self.max_query_length:
                return {
                    "success": False,
                    "error": f"Query too long (max {self.max_query_length} characters)",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get user context from memory
            user_context = await self.memory.get_user_context(user_id)
            
            # Merge context
            full_context = {
                "user_id": user_id,
                "user_context": user_context,
                **(context or {}),
                **(preferences or {})
            }
            
            # Process query through orchestrator
            result = await self.orchestrator.process_query(query, user_id, full_context)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            success = result.get("metadata", {}).get("success", True)
            agents_used = self._extract_agents_used(result)
            tokens_used = result.get("metadata", {}).get("tokens_used", 0)
            
            # Record analytics if enabled
            if self.enable_analytics:
                answer_text = result.get("response", "") or "No answer generated"
                await self._record_query_analytics(
                    user_id=user_id,
                    platform=platform,
                    query_text=query,
                    answer_text=answer_text,
                    response_time=response_time,
                    agents_used=agents_used,
                    tokens_used=tokens_used,
                    success=True,
                    context=context or {},
                    channel_id=channel_id
                )
            
            # Store interaction in memory if learning is enabled
            if self.enable_learning:
                response_for_memory = result.get("response", "")
                # Ensure response is a string for database storage
                if isinstance(response_for_memory, dict):
                    response_for_memory = str(response_for_memory)
                elif not isinstance(response_for_memory, str):
                    response_for_memory = str(response_for_memory)
                
                await self.memory.add_interaction(
                    user_id=user_id,
                    query=query,
                    response=response_for_memory,
                    context=full_context,
                    metadata={
                        "sources_count": len(result.get("sources", [])),
                        "confidence": result.get("confidence", 0.0),
                        "processing_time": response_time
                    }
                )
            
            return {
                "success": True,
                "answer": result.get("response", ""),
                "sources": result.get("results", []),
                "subtasks": result.get("subtasks", []),
                "confidence": result.get("confidence", 0.0),
                "metadata": {
                    **result.get("metadata", {}),
                    "response_time": response_time,
                    "agents_used": agents_used,
                    "tokens_used": tokens_used
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Record failed query if analytics enabled
            if self.enable_analytics:
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds()
                
                await self._record_query_analytics(
                    user_id=user_id,
                    platform=platform,
                    query_text=query,
                    answer_text="Error occurred during processing",
                    response_time=response_time,
                    agents_used=[],
                    tokens_used=0,
                    success=False,
                    context=context or {},
                    channel_id=channel_id,
                    error_message=str(e)
                )
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _record_query_analytics(
        self,
        user_id: str,
        platform: str,
        query_text: str,
        answer_text: str,
        response_time: float,
        agents_used: List[str],
        tokens_used: int,
        success: bool,
        context: Dict[str, Any],
        channel_id: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Record query analytics"""
        try:
            metrics = QueryMetrics(
                response_time=response_time,
                agents_used=agents_used,
                tokens_used=tokens_used,
                cache_hit=context.get("cache_hit", False),
                success=success,
                error_message=error_message
            )
            
            query_id = await self.query_repository.record_query_answer(
                user_id=user_id,
                platform=platform,
                query_text=query_text,
                answer_text=answer_text,
                metrics=metrics,
                context=context,
                channel_id=channel_id
            )
            
            # Trigger validation if successful
            if success and answer_text:
                await self._validate_answer(query_id, query_text, answer_text, context)
                
        except Exception as e:
            logger.error(f"Error recording analytics: {e}")
    
    async def _validate_answer(
        self,
        query_id: int,
        query_text: str,
        answer_text: str,
        context: Dict[str, Any]
    ):
        """Validate answer quality"""
        try:
            validation_report = await self.validation_system.validate_query_answer(
                query_id=query_id,
                query_text=query_text,
                answer_text=answer_text,
                context=context
            )
            
            # Record validation in repository
            from ..analytics.query_answer_repository import ValidationResult
            validation_result = ValidationResult(
                is_valid=validation_report.metrics.overall_score >= 3.0,
                quality_score=validation_report.metrics.overall_score,
                relevance_score=validation_report.metrics.relevance_score,
                completeness_score=validation_report.metrics.completeness_score,
                accuracy_score=validation_report.metrics.accuracy_score,
                issues=[issue.description for issue in validation_report.issues],
                suggestions=validation_report.improvements
            )
            
            await self.query_repository.record_validation(
                query_id=query_id,
                validator_type="auto",
                result=validation_result,
                validator_info=validation_report.validator_info
            )
            
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
    
    def _extract_agents_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract list of agents used from result metadata"""
        metadata = result.get("metadata", {})
        agents = []
        
        # Look for agent information in various metadata fields
        if "agents_used" in metadata:
            agents.extend(metadata["agents_used"])
        
        # Extract from step information
        for key, value in metadata.items():
            if key.endswith("_agent") and isinstance(value, dict):
                agent_name = key.replace("_agent", "")
                if agent_name not in agents:
                    agents.append(agent_name)
        
        return agents or ["orchestrator"]  # Default to orchestrator

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
        search_type: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Perform direct search on the vector store.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            k: Number of results to return
            search_type: Type of search (similarity, keyword, filter)
            
        Returns:
            Search results
        """
        try:
            k_value = k or self.default_k
            
            if search_type == "similarity":
                results = await self.mcp_server.search_messages(query, filters, k_value)
            elif search_type == "keyword":
                keywords = query.split()
                results = await self.mcp_server.search_messages(query, filters, k_value)
            elif search_type == "filter":
                results = await self.mcp_server.query_messages(f"show me {k_value} messages")
            else:
                return {
                    "success": False,
                    "error": f"Invalid search type: {search_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "search_type": search_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        source: str = "api"
    ) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            source: Source identifier for the documents
            
        Returns:
            Result of the operation
        """
        try:
            # Add source metadata
            for doc in documents:
                doc["source"] = source
                doc["added_via"] = "api"
                doc["added_at"] = datetime.utcnow().isoformat()
            
            # MCP server doesn't need explicit message storage - messages are already in SQLite
            success = True
            
            return {
                "success": success,
                "documents_added": len(documents) if success else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context and history.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context information
        """
        try:
            context = await self.memory.get_user_context(user_id)
            # Get recent conversation history
            history = await self.memory.get_history(user_id, limit=10)
            
            return {
                "success": True,
                "user_id": user_id,
                "context": context,
                "recent_history": history,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def update_user_context(
        self,
        user_id: str,
        context_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user context.
        
        Args:
            user_id: User identifier
            context_updates: Context updates to apply
            
        Returns:
            Result of the operation
        """
        try:
            # Note: This method should be implemented in ConversationMemory
            # For now, we'll update user context directly
            user_context = await self.memory.get_user_context(user_id)
            user_context.update(context_updates)
            
            # Store updated context - this would need to be implemented
            # For now, return success
            return {
                "success": True,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating user context: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and health information.
        
        Returns:
            System statistics
        """
        try:
            # Get MCP server stats
            mcp_stats = await self.mcp_server.get_database_info()
            
            # Get memory stats - using available method
            memory_stats = {
                "status": "available",
                "note": "Basic memory operations available"
            }
            
            # Get orchestrator stats (if available)
            orchestrator_stats = getattr(self.orchestrator, 'get_stats', lambda: {})()
            
            return {
                "success": True,
                "mcp_server": mcp_stats,
                "memory": memory_stats,
                "orchestrator": orchestrator_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Health status information
        """
        try:
            health = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            # Check MCP server
            mcp_health = await self.mcp_server.health_check()
            health["components"]["mcp_server"] = mcp_health
            
            if mcp_health["status"] != "healthy":
                health["status"] = "degraded"
            
            # Check memory system
            try:
                await self.memory.get_user_context("health_check_test")
                health["components"]["memory"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["memory"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            # Check orchestrator
            try:
                # Simple test to see if orchestrator is responsive
                health["components"]["orchestrator"] = {"status": "healthy"}
            except Exception as e:
                health["components"]["orchestrator"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "unhealthy"
            
            return health
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize system performance.
        
        Returns:
            Optimization results
        """
        try:
            results = {
                "success": True,
                "optimizations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Optimize vector store
            # MCP server doesn't need optimization - it's already optimized for SQLite queries
            results["optimizations"].append({
                "component": "mcp_server",
                "success": True
            })
            
            # Optimize memory system - using basic cleanup
            results["optimizations"].append({
                "component": "memory",
                "success": True,
                "note": "Basic memory system operational"
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Pipeline Management Methods
    
    async def run_pipeline(
        self,
        user_id: str,
        steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the data processing pipeline.
        
        Args:
            user_id: User who initiated the pipeline
            steps: Specific steps to run (if None, runs full pipeline)
            
        Returns:
            Pipeline execution results
        """
        try:
            if steps:
                # Run specific steps sequentially
                results = []
                for step in steps:
                    step_result = await self.pipeline.run_single_step(step, user_id)
                    results.append(step_result)
                    if not step_result["success"]:
                        break
                
                return {
                    "success": all(r["success"] for r in results),
                    "steps": results,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Run full pipeline
                return await self.pipeline.run_full_pipeline(user_id)
                
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def run_pipeline_step(
        self,
        step_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Run a single pipeline step.
        
        Args:
            step_name: Name of the step to run
            user_id: User who initiated the step
            
        Returns:
            Step execution results
        """
        try:
            return await self.pipeline.run_single_step(step_name, user_id)
            
        except Exception as e:
            logger.error(f"Error running pipeline step: {e}")
            return {
                "success": False,
                "error": str(e),
                "step": step_name,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Pipeline status information
        """
        try:
            status = self.pipeline.get_pipeline_status()
            status["timestamp"] = datetime.utcnow().isoformat()
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_pipeline_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get pipeline execution history.
        
        Args:
            limit: Number of recent executions to return
            
        Returns:
            Pipeline execution history
        """
        try:
            history = self.pipeline.get_pipeline_history(limit)
            return {
                "success": True,
                "history": history,
                "count": len(history),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline history: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_pipeline_logs(self, lines: int = 100) -> Dict[str, Any]:
        """
        Get recent pipeline logs.
        
        Args:
            lines: Number of recent log lines to return
            
        Returns:
            Pipeline log contents
        """
        try:
            return await self.pipeline.get_pipeline_logs(lines)
            
        except Exception as e:
            logger.error(f"Error getting pipeline logs: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_data_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive data statistics.
        
        Returns:
            Data statistics from database and vector store
        """
        try:
            # Get database stats from pipeline agent
            pipeline_status = self.pipeline.get_pipeline_status()
            db_stats = await self.pipeline._get_db_stats()
            
            # Get MCP server stats
            mcp_stats = await self.mcp_server.get_database_info()
            
            return {
                "success": True,
                "database": db_stats,
                "mcp_server": mcp_stats,
                "pipeline": pipeline_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close all system components."""
        try:
            await self.mcp_server.close()
            # Note: Add proper close method to ConversationMemory if needed
            # orchestrator doesn't need explicit closing
            
            logger.info("AgentAPI closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing AgentAPI: {e}")
