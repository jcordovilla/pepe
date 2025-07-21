"""
Bot Runner

Executes test queries against the Discord bot and collects responses.
Handles sequential processing, error scenarios, and response collection.
"""

import time
from typing import List, Any, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import asyncio

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

class BotRunner:
    """
    Executes test queries against the bot in-process and collects responses.
    Handles sequential processing, error scenarios, and timing.
    """
    def __init__(self, bot_api_endpoint: str = None):
        # Ignore bot_api_endpoint, use in-process AgentAPI
        config = get_modernized_config()
        self.agent_api = AgentAPI(config)

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
        return responses

    async def _execute_single_query(self, query: Any) -> BotResponse:
        start_time = time.time()
        try:
            # Prepare context (simulate Discord/CLI context)
            context = {
                "platform": "cli",
                "channel_id": "test_channel_456",
                "timestamp": datetime.utcnow().isoformat(),
                "test_query_id": query.id,
                "category": query.category,
                "complexity": query.complexity,
                "edge_case": query.edge_case
            }
            # Call AgentAPI directly
            result = await self.agent_api.query(
                query=query.query,
                user_id="test_user_123",
                context=context
            )
            response_text = result.get("answer", "")
            response_time = time.time() - start_time
            response = BotResponse(
                query_id=query.id,
                query=query.query,
                response=response_text,
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat(),
                success=result.get("success", False),
                metadata={
                    "query_category": query.category,
                    "query_complexity": query.complexity,
                    "response_length": len(response_text),
                    "words_per_second": len(response_text.split()) / response_time if response_time > 0 else 0,
                    "agent_metadata": result.get("metadata", {})
                }
            )
            logger.info(f"Query {query.id} completed in {response_time:.2f}s")
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
                metadata={}
            )
    
    async def run_error_scenarios(self) -> List[BotResponse]:
        """
        Run additional error scenario tests.
        
        Returns:
            List of BotResponse objects for error scenarios
        """
        logger.info("Running error scenario tests...")
        
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
        # This method will need to be refactored to work with the new BotResponse structure
        # and potentially the AgentAPI's response format.
        # For now, returning a placeholder.
        return {
            "total_queries": 0, # Placeholder
            "successful_queries": 0, # Placeholder
            "failed_queries": 0, # Placeholder
            "success_rate": 0.0, # Placeholder
            "performance_metrics": {}, # Placeholder
            "error_summary": {} # Placeholder
        }
    
    def _get_common_errors(self, failed_responses: List[BotResponse]) -> Dict[str, int]:
        """Get common error patterns from failed responses."""
        error_counts = {}
        
        for response in failed_responses:
            if response.error_message:
                # Extract error type from error message
                error_type = "Unknown"
                if "timeout" in response.error_message.lower():
                    error_type = "Timeout"
                elif "api" in response.error_message.lower():
                    error_type = "API Error"
                elif "connection" in response.error_message.lower():
                    error_type = "Connection Error"
                elif "rate limit" in response.error_message.lower():
                    error_type = "Rate Limit"
                
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
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
                "execution_summary": self.get_execution_summary()
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