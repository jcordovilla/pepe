"""
Bot Runner

Executes test queries against the Discord bot and collects responses.
Handles sequential processing, error scenarios, and response collection.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BotResponse:
    """Represents a bot response to a test query."""
    query_id: int
    query: str
    response: str
    response_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BotRunner:
    """
    Executes test queries against the Discord bot and collects responses.
    
    Features:
    - Sequential query processing
    - Error handling and recovery
    - Response time measurement
    - Response metadata collection
    """
    
    def __init__(self, bot_api_endpoint: str = "http://localhost:8000"):
        self.bot_api_endpoint = bot_api_endpoint
        self.responses = []
        self.session = None
        
        logger.info(f"BotRunner initialized with endpoint: {bot_api_endpoint}")
    
    async def run_queries(self, queries: List[Any]) -> List[BotResponse]:
        """
        Execute all test queries against the bot.
        
        Args:
            queries: List of TestQuery objects
            
        Returns:
            List of BotResponse objects with responses and metadata
        """
        logger.info(f"Starting execution of {len(queries)} queries...")
        
        self.responses = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Executing query {i}/{len(queries)}: {query.query[:50]}...")
            
            try:
                response = await self._execute_single_query(query)
                self.responses.append(response)
                
                # Add delay between queries to avoid rate limiting
                if i < len(queries):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error executing query {query.id}: {e}")
                error_response = BotResponse(
                    query_id=query.id,
                    query=query.query,
                    response="",
                    response_time=0.0,
                    timestamp=datetime.utcnow().isoformat(),
                    success=False,
                    error_message=str(e)
                )
                self.responses.append(error_response)
        
        logger.info(f"Completed execution of {len(queries)} queries")
        return self.responses
    
    async def _execute_single_query(self, query: Any) -> BotResponse:
        """
        Execute a single query against the bot.
        
        Args:
            query: TestQuery object
            
        Returns:
            BotResponse object with response and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "query": query.query,
                "user_id": "test_user_123",
                "channel_id": "test_channel_456",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "test_query_id": query.id,
                    "category": query.category,
                    "complexity": query.complexity,
                    "edge_case": query.edge_case
                }
            }
            
            # Make the API call to the bot
            response_text = await self._call_bot_api(payload)
            
            response_time = time.time() - start_time
            
            # Create response object
            response = BotResponse(
                query_id=query.id,
                query=query.query,
                response=response_text,
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                metadata={
                    "query_category": query.category,
                    "query_complexity": query.complexity,
                    "response_length": len(response_text),
                    "words_per_second": len(response_text.split()) / response_time if response_time > 0 else 0
                }
            )
            
            logger.info(f"Query {query.id} completed in {response_time:.2f}s")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error in query {query.id}: {e}")
            
            return BotResponse(
                query_id=query.id,
                query=query.query,
                response="",
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    async def _call_bot_api(self, payload: Dict[str, Any]) -> str:
        """
        Make API call to the bot endpoint.
        
        Args:
            payload: Request payload
            
        Returns:
            Response text from the bot
        """
        try:
            import aiohttp
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                f"{self.bot_api_endpoint}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception("Request timeout")
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
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
        if not self.responses:
            return {}
        
        successful_responses = [r for r in self.responses if r.success]
        failed_responses = [r for r in self.responses if not r.success]
        
        if successful_responses:
            response_times = [r.response_time for r in successful_responses]
            response_lengths = [r.response_length for r in successful_responses if r.metadata]
            
            summary = {
                "total_queries": len(self.responses),
                "successful_queries": len(successful_responses),
                "failed_queries": len(failed_responses),
                "success_rate": (len(successful_responses) / len(self.responses)) * 100,
                "performance_metrics": {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0
                },
                "error_summary": {
                    "total_errors": len(failed_responses),
                    "common_errors": self._get_common_errors(failed_responses)
                }
            }
        else:
            summary = {
                "total_queries": len(self.responses),
                "successful_queries": 0,
                "failed_queries": len(failed_responses),
                "success_rate": 0.0,
                "error_summary": {
                    "total_errors": len(failed_responses),
                    "common_errors": self._get_common_errors(failed_responses)
                }
            }
        
        return summary
    
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
        if self.session:
            await self.session.close()
            self.session = None 