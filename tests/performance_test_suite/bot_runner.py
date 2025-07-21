"""
Bot Runner

Executes test queries against the Discord bot and collects responses.
Handles sequential processing, error scenarios, and response collection.
Uses the exact same AgentAPI.query() method as Discord bot and CLI.
"""

import json
import time
import sqlite3
from typing import List, Any, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import asyncio
from pathlib import Path

from agentic.config.modernized_config import get_modernized_config
from agentic.interfaces.agent_api import AgentAPI

logger = logging.getLogger(__name__)

@dataclass
class BotResponse:
    query_id: int
    query: str
    response: str
    response_time: float
    timestamp: str
    success: bool
    metadata: Dict[str, Any]


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class DatabaseError(Exception):
    """Raised when database access or validation fails."""
    pass

class BotRunner:
    """
    Executes test queries against the bot using the exact same AgentAPI.query() method
    as the Discord bot and CLI, with real data from SQLite and vector store.
    """
    def __init__(self, bot_api_endpoint: str = None):
        # Use the exact same configuration as production Discord bot
        config = get_modernized_config()
        
        # Validate configuration alignment
        self._validate_config_alignment(config)
        
        self.agent_api = AgentAPI(config)
        self.responses = []
        
        # Database path for real data - same as production
        self.db_path = Path("data/discord_messages.db")
        
        # Validate database exists and is accessible
        self._validate_database_access()
        
        logger.info("BotRunner initialized with production configuration and real database access")

    def _validate_config_alignment(self, config: Dict[str, Any]):
        """Validate that test configuration matches production."""
        required_keys = [
            "llm.endpoint",
            "llm.model", 
            "data.vector_config.persist_directory",
            "data.vector_config.collection_name",
            "data.vector_config.embedding_model"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self._get_nested_value(config, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ConfigError(f"Missing required configuration keys: {missing_keys}")
        
        logger.info("Configuration alignment validated - using production settings")

    def _validate_database_access(self):
        """Validate database access and schema."""
        try:
            if not self.db_path.exists():
                raise DatabaseError(f"Database not found: {self.db_path}")
            
            # Test database connection
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if messages table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
            if not cursor.fetchone():
                raise DatabaseError(f"Messages table not found in {self.db_path}")
            
            # Check message count
            cursor.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            logger.info(f"Database validated: {count} messages found")
            
            conn.close()
            
        except sqlite3.Error as e:
            raise DatabaseError(f"Database access error: {e}")
        except Exception as e:
            raise DatabaseError(f"Database validation failed: {e}")

    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

    async def run_queries(self, queries: List[Any]) -> List[BotResponse]:
        responses = []
        for query in queries:
            try:
                if not query or not hasattr(query, 'query'):
                    logger.warning(f"Skipping malformed query: {query}")
                    continue
                response = await self._execute_single_query(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error executing query {getattr(query, 'id', '?')}: {e}")
        
        self.responses = responses
        return responses

    async def run_queries_parallel(self, queries: List[Any], max_concurrent: int = 5) -> List[BotResponse]:
        """
        Execute queries in parallel with proper resource management.
        
        Args:
            queries: List of queries to execute
            max_concurrent: Maximum number of concurrent queries
            
        Returns:
            List of bot responses
        """
        logger.info(f"Running {len(queries)} queries in parallel (max {max_concurrent} concurrent)")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(query: Any) -> BotResponse:
            """Execute a single query with semaphore control."""
            async with semaphore:
                try:
                    return await self._execute_single_query(query)
                except Exception as e:
                    logger.error(f"Error executing query {getattr(query, 'id', '?')}: {e}")
                    # Return error response
                    return BotResponse(
                        query_id=getattr(query, 'id', 0),
                        query=getattr(query, 'query', ''),
                        response=f"Error: {str(e)}",
                        response_time=0.0,
                        timestamp=datetime.utcnow().isoformat(),
                        success=False,
                        metadata={"error": str(e)}
                    )
        
        # Create tasks for all queries
        tasks = [execute_with_semaphore(query) for query in queries]
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Query {i} failed with exception: {response}")
                final_responses.append(BotResponse(
                    query_id=getattr(queries[i], 'id', i),
                    query=getattr(queries[i], 'query', ''),
                    response=f"Exception: {str(response)}",
                    response_time=0.0,
                    timestamp=datetime.utcnow().isoformat(),
                    success=False,
                    metadata={"exception": str(response)}
                ))
            else:
                final_responses.append(response)
        
        self.responses = final_responses
        logger.info(f"Parallel execution completed: {len(final_responses)} responses")
        return final_responses

    async def _execute_single_query(self, query: Any) -> BotResponse:
        start_time = time.time()
        try:
            # Get real context data from SQLite database
            real_context = await self._get_real_context_from_database()
            
            # Use the exact same AgentAPI.query() method as Discord bot
            result = await self.agent_api.query(
                query=query.query,
                user_id=real_context["user_id"],
                context=real_context
            )
            
            # Handle response exactly like Discord bot does
            if result is None:
                result = {"success": False, "answer": "No response from agentic system", "metadata": {}}
            
            response_text = result.get("answer", "") or ""
            if not response_text and result.get("success", False):
                response_text = "Empty response from agentic system"
            
            response_time = time.time() - start_time
            response = BotResponse(
                query_id=query.id,
                query=query.query,
                response=response_text,
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat(),
                success=result.get("success", False) and len(response_text) > 0,
                metadata={
                    "query_category": query.category,
                    "query_complexity": query.complexity,
                    "response_length": len(response_text),
                    "words_per_second": len(response_text.split()) / response_time if response_time > 0 else 0,
                    "agent_metadata": result.get("metadata", {}),
                    "real_context_used": True,
                    "platform": real_context.get("platform", "unknown")
                }
            )
            logger.info(f"Query {query.id} completed in {response_time:.2f}s using real context")
            return response
        except Exception as e:
            logger.error(f"Error in query {query.id}: {e}")
            response_time = time.time() - start_time
            return BotResponse(
                query_id=query.id,
                query=query.query,
                response=str(e),
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat(),
                success=False,
                metadata={"error": str(e)}
            )

    async def _get_real_context_from_database(self) -> Dict[str, Any]:
        """
        Get real context data from SQLite database, exactly like Discord bot would.
        """
        try:
            if not self.db_path.exists():
                logger.warning("Database not found, using fallback context")
                return self._get_fallback_context()
            
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get real channel and user data from recent messages
            cursor.execute("""
                SELECT DISTINCT 
                    channel_id, 
                    channel_name, 
                    author_id, 
                    author_username,
                    guild_id
                FROM messages 
                WHERE channel_name NOT LIKE '%test%'
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                channel_id, channel_name, author_id, author_username, guild_id = row
                
                # Create real context exactly like Discord bot
                real_context = {
                    "platform": "discord",  # Same as Discord bot
                    "channel_id": int(channel_id) if channel_id else 123456789,
                    "channel_name": channel_name or "general",
                    "guild_id": int(guild_id) if guild_id else 987654321,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": str(author_id) if author_id else "test_user_123",
                    "username": author_username or "TestUser"
                }
                
                logger.info(f"Using real context: {channel_name} by {author_username}")
                return real_context
            else:
                logger.warning("No real data found in database, using fallback context")
                return self._get_fallback_context()
                
        except Exception as e:
            logger.error(f"Error getting real context: {e}")
            return self._get_fallback_context()

    def _get_fallback_context(self) -> Dict[str, Any]:
        """
        Fallback context when database is not available.
        Still uses the same structure as Discord bot.
        """
        return {
            "platform": "discord",
            "channel_id": 123456789,
            "channel_name": "general",
            "guild_id": 987654321,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "test_user_123",
            "username": "TestUser"
        }

    async def run_error_scenarios(self) -> List[BotResponse]:
        """
        Run additional error scenario tests using real AgentAPI.
        """
        logger.info("Running error scenario tests with real AgentAPI...")
        
        error_queries = [
            {
                "id": "error_001",
                "query": "",  # Empty query
                "description": "Empty query test"
            },
            {
                "id": "error_002", 
                "query": "a" * 10000,  # Very long query
                "description": "Very long query test"
            },
            {
                "id": "error_003",
                "query": "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
                "description": "SQL injection test"
            },
            {
                "id": "error_004",
                "query": "🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴",  # Emoji spam
                "description": "Emoji spam test"
            },
            {
                "id": "error_005",
                "query": "http://malicious-site.com/exploit",  # Malicious URL
                "description": "Malicious URL test"
            }
        ]
        
        error_responses = []
        
        for error_query in error_queries:
            try:
                # Create a mock TestQuery object
                class MockQuery:
                    def __init__(self, data):
                        self.id = data["id"]
                        self.query = data["query"]
                        self.category = "error_test"
                        self.complexity = "simple"
                        self.edge_case = True
                
                mock_query = MockQuery(error_query)
                response = await self._execute_single_query(mock_query)
                error_responses.append(response)
                
            except Exception as e:
                logger.error(f"Error in error scenario {error_query['id']}: {e}")
        
        logger.info(f"Completed {len(error_responses)} error scenario tests")
        return error_responses
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution results."""
        if not self.responses:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0.0,
                "performance_metrics": {},
                "error_summary": {}
            }
        
        successful = [r for r in self.responses if r.success]
        failed = [r for r in self.responses if not r.success]
        
        response_times = [r.response_time for r in self.responses if r.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_queries": len(self.responses),
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "success_rate": (len(successful) / len(self.responses)) * 100 if self.responses else 0,
            "performance_metrics": {
                "average_response_time": avg_response_time,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0
            },
            "error_summary": {
                "total_errors": len(failed),
                "error_rate": (len(failed) / len(self.responses)) * 100 if self.responses else 0
            }
        }
    
    def save_responses(self, filename: Optional[str] = None) -> str:
        """Save bot responses to a JSON file."""
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/performance_test_suite/data/bot_responses_{timestamp}.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert responses to serializable format
        responses_data = []
        for response in self.responses:
            response_dict = asdict(response)
            responses_data.append(response_dict)
        
        data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_responses": len(self.responses),
                "execution_summary": self.get_execution_summary(),
                "real_data_used": True,
                "agent_api_version": "same_as_discord_bot"
            },
            "responses": responses_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Bot responses saved to: {filename}")
        return filename
    
    async def cleanup(self):
        """Clean up resources."""
        # No explicit session to close here as AgentAPI is in-process
        pass 