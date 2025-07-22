"""
Bot Runner

Executes test queries against the Discord bot and collects responses.
Handles sequential processing, error scenarios, and response collection.
Uses the exact same AgentAPI.query() method as Discord bot and CLI.
Updated for v2 agentic architecture with comprehensive progress reporting.
"""

import json
import time
import sqlite3
from typing import List, Any, Dict, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
import asyncio
from pathlib import Path
import sys
from tqdm import tqdm

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
    agent_used: Optional[str] = None
    routing_info: Optional[Dict[str, Any]] = None
    validation_passed: Optional[bool] = None


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
    Updated for v2 agentic architecture with enhanced progress tracking.
    """
    def __init__(self, bot_api_endpoint: str = None):
        # Use the exact same configuration as production Discord bot
        config = get_modernized_config()
        
        # Validate configuration alignment
        self._validate_config_alignment(config)
        
        self.agent_api = AgentAPI(config)
        self.responses = []
        self.initialized = False
        
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
                raise DatabaseError("Messages table not found in database")
            
            # Check table structure
            cursor.execute("PRAGMA table_info(messages)")
            columns = [row[1] for row in cursor.fetchall()]
            required_columns = ['message_id', 'content', 'author_id', 'channel_id', 'timestamp']
            
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                raise DatabaseError(f"Missing required columns: {missing_columns}")
            
            conn.close()
            logger.info("Database validation passed")
            
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

    async def initialize(self):
        """Initialize the bot runner for v2 agentic architecture."""
        if self.initialized:
            return
        
        try:
            # Initialize the agent API
            await self.agent_api.initialize()
            self.initialized = True
            logger.info("BotRunner initialized for v2 agentic architecture")
        except Exception as e:
            logger.error(f"Failed to initialize BotRunner: {e}")
            raise

    async def run_queries_with_progress(self, queries: List[Any]) -> List[BotResponse]:
        """
        Execute queries with comprehensive progress reporting.
        
        Args:
            queries: List of test queries to execute
            
        Returns:
            List of BotResponse objects with detailed metadata
        """
        if not self.initialized:
            await self.initialize()
        
        print_info("🤖 Starting query execution with v2 agents...")
        print_info(f"   Total queries: {len(queries)}")
        
        responses = []
        successful_queries = 0
        failed_queries = 0
        
        # Create progress bar
        with tqdm(total=len(queries), desc="Executing queries", unit="query") as pbar:
            for i, query in enumerate(queries, 1):
                try:
                    # Execute single query with detailed progress
                    response = await self._execute_single_query_with_progress(query, i, len(queries))
                    responses.append(response)
                    
                    if response.success:
                        successful_queries += 1
                    else:
                        failed_queries += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Success': f"{successful_queries}/{i}",
                        'Failed': failed_queries,
                        'Agent': response.agent_used or 'unknown',
                        'Time': f"{response.response_time:.2f}s"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Query {i} failed: {e}")
                    failed_queries += 1
                    
                    # Create error response
                    error_response = BotResponse(
                        query_id=query.id,
                        query=query.query,
                        response=f"Error: {str(e)}",
                        response_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        success=False,
                        metadata={"error": str(e)},
                        agent_used="error"
                    )
                    responses.append(error_response)
                    
                    pbar.set_postfix({
                        'Success': f"{successful_queries}/{i}",
                        'Failed': failed_queries,
                        'Agent': 'error',
                        'Time': '0.00s'
                    })
                    pbar.update(1)
        
        # Print execution summary
        print_success(f"✅ Query execution completed")
        print_info(f"   - Successful: {successful_queries}/{len(queries)} ({successful_queries/len(queries)*100:.1f}%)")
        print_info(f"   - Failed: {failed_queries}/{len(queries)} ({failed_queries/len(queries)*100:.1f}%)")
        
        # Analyze agent usage
        agent_usage = self._analyze_agent_usage(responses)
        print_info("   - Agent usage:")
        for agent, count in agent_usage.items():
            print_info(f"     • {agent}: {count} queries")
        
        return responses

    async def _execute_single_query_with_progress(self, query: Any, query_num: int, total_queries: int) -> BotResponse:
        """
        Execute a single query with detailed progress reporting.
        
        Args:
            query: Test query to execute
            query_num: Current query number
            total_queries: Total number of queries
            
        Returns:
            BotResponse with detailed metadata
        """
        start_time = time.time()
        
        # Print query details
        print_progress("Query", query_num, total_queries, f"Executing: {query.query[:50]}...")
        
        try:
            # Get real context from database
            context = await self._get_real_context_from_database()
            
            # Execute query using AgentAPI
            result = await self.agent_api.query(
                query=query.query,
                user_id="test_user",
                context=context
            )
            
            response_time = time.time() - start_time
            
            # Extract agent information from result
            agent_used = result.get("metadata", {}).get("agent_used", "unknown")
            routing_info = result.get("metadata", {}).get("routing_result")
            validation_passed = result.get("metadata", {}).get("validation_passes", True)
            
            # Create response object
            response = BotResponse(
                query_id=query.id,
                query=query.query,
                response=result.get("response", "No response"),
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                success=result.get("metadata", {}).get("success", True),
                metadata=result.get("metadata", {}),
                agent_used=agent_used,
                routing_info=routing_info,
                validation_passed=validation_passed
            )
            
            # Print success/failure status
            if response.success:
                print_success(f"   ✅ Query {query_num} completed successfully ({response_time:.2f}s)")
                print_info(f"      Agent: {agent_used}")
                if validation_passed is not None:
                    validation_status = "✅" if validation_passed else "⚠️"
                    print_info(f"      Validation: {validation_status}")
            else:
                print_error(f"   ❌ Query {query_num} failed ({response_time:.2f}s)")
                print_info(f"      Agent: {agent_used}")
                print_info(f"      Error: {result.get('metadata', {}).get('error', 'Unknown error')}")
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Query execution failed: {e}")
            
            # Create error response
            return BotResponse(
                query_id=query.id,
                query=query.query,
                response=f"Execution error: {str(e)}",
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                success=False,
                metadata={"error": str(e), "error_type": type(e).__name__},
                agent_used="error"
            )

    async def _get_real_context_from_database(self) -> Dict[str, Any]:
        """Get real context from the database for authentic testing."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent messages for context
            cursor.execute("""
                SELECT content, author_id, channel_id, timestamp 
                FROM messages 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            
            recent_messages = cursor.fetchall()
            
            # Get channel information
            cursor.execute("""
                SELECT DISTINCT channel_id, channel_name 
                FROM messages 
                WHERE channel_name IS NOT NULL 
                LIMIT 5
            """)
            
            channels = cursor.fetchall()
            
            conn.close()
            
            # Build context
            context = {
                "recent_messages": [
                    {
                        "content": msg[0],
                        "author_id": msg[1],
                        "channel_id": msg[2],
                        "timestamp": msg[3]
                    }
                    for msg in recent_messages
                ],
                "channels": [
                    {
                        "channel_id": ch[0],
                        "channel_name": ch[1]
                    }
                    for ch in channels
                ],
                "test_mode": True
            }
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get real context: {e}")
            return self._get_fallback_context()

    def _get_fallback_context(self) -> Dict[str, Any]:
        """Get fallback context when database access fails."""
        return {
            "recent_messages": [],
            "channels": [],
            "test_mode": True,
            "fallback_context": True
        }

    def _analyze_agent_usage(self, responses: List[BotResponse]) -> Dict[str, int]:
        """Analyze which agents were used during query execution."""
        agent_usage = {}
        for response in responses:
            agent = response.agent_used or "unknown"
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        return agent_usage

    async def run_error_scenarios(self) -> List[BotResponse]:
        """Run additional error scenario tests."""
        print_info("🧪 Running error scenario tests...")
        
        error_queries = [
            {
                "id": "error_1",
                "query": "",  # Empty query
                "category": "error_testing",
                "subcategory": "empty_query"
            },
            {
                "id": "error_2", 
                "query": "x" * 10000,  # Very long query
                "category": "error_testing",
                "subcategory": "long_query"
            },
            {
                "id": "error_3",
                "query": "SELECT * FROM users; DROP TABLE messages;",  # SQL injection attempt
                "category": "error_testing", 
                "subcategory": "sql_injection"
            }
        ]
        
        responses = []
        for i, query_data in enumerate(error_queries, 1):
            try:
                class MockQuery:
                    def __init__(self, data):
                        self.id = data["id"]
                        self.query = data["query"]
                        self.category = data["category"]
                        self.subcategory = data["subcategory"]
                
                mock_query = MockQuery(query_data)
                response = await self._execute_single_query_with_progress(mock_query, i, len(error_queries))
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error scenario {i} failed: {e}")
                responses.append(BotResponse(
                    query_id=query_data["id"],
                    query=query_data["query"],
                    response=f"Error scenario failed: {str(e)}",
                    response_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    metadata={"error": str(e), "scenario": query_data["subcategory"]},
                    agent_used="error"
                ))
        
        print_success(f"✅ Error scenarios completed: {len(responses)} tests")
        return responses

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution results."""
        if not self.responses:
            return {"status": "No queries executed"}
        
        total_queries = len(self.responses)
        successful_queries = sum(1 for r in self.responses if r.success)
        failed_queries = total_queries - successful_queries
        
        # Calculate timing statistics
        response_times = [r.response_time for r in self.responses if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Analyze agent usage
        agent_usage = self._analyze_agent_usage(self.responses)
        
        # Analyze validation results
        validation_results = [r.validation_passed for r in self.responses if r.validation_passed is not None]
        validation_pass_rate = (sum(validation_results) / len(validation_results) * 100) if validation_results else 0
        
        return {
            "status": "completed",
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "timing": {
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "agent_usage": agent_usage,
            "validation": {
                "pass_rate": validation_pass_rate,
                "total_validated": len(validation_results)
            }
        }

    def save_responses(self, filename: Optional[str] = None) -> str:
        """Save responses to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bot_responses_{timestamp}.json"
        
        # Convert responses to serializable format
        serializable_responses = []
        for response in self.responses:
            response_dict = response.__dict__.copy()
            serializable_responses.append(response_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_responses, f, indent=2, default=str)
        
        logger.info(f"Responses saved to: {filename}")
        return filename

    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self.agent_api, 'cleanup'):
                await self.agent_api.cleanup()
            logger.info("BotRunner cleanup completed")
        except Exception as e:
            logger.error(f"Error during BotRunner cleanup: {e}")


# Color utility functions
def print_info(msg):
    print(f"\033[94m{msg}\033[0m")
    sys.stdout.flush()

def print_success(msg):
    print(f"\033[92m{msg}\033[0m")
    sys.stdout.flush()

def print_error(msg):
    print(f"\033[91m{msg}\033[0m")
    sys.stdout.flush()

def print_progress(phase: str, current: int, total: int, description: str = ""):
    """Print progress with timestamp and percentage."""
    from datetime import datetime
    percentage = (current / total) * 100 if total > 0 else 0
    timestamp = datetime.now().strftime("%H:%M:%S")
    progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    print(f"\033[96m[{timestamp}] {phase}: [{progress_bar}] {current}/{total} ({percentage:.1f}%) {description}\033[0m")
    sys.stdout.flush() 