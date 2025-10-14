"""
Enhanced Bot Runner

Advanced bot runner with parallel execution, performance baselines, 
regression testing, and resource monitoring capabilities.
"""

import asyncio
import json
import time
import sqlite3
import psutil
import statistics
from typing import List, Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

from agentic.config.modernized_config import get_modernized_config
from agentic.interfaces.agent_api import AgentAPI

logger = logging.getLogger(__name__)

@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    name: str
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    success_rate: float
    max_memory_mb: float
    max_cpu_percent: float
    timestamp: str
    query_count: int
    agent_usage: Dict[str, int]

@dataclass
class ResourceMetrics:
    """System resource metrics during testing."""
    timestamp: str
    cpu_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float

@dataclass
class RegressionTestResult:
    """Result of regression testing against baseline."""
    test_name: str
    baseline_name: str
    passed: bool
    performance_degradation: float
    memory_increase: float
    cpu_increase: float
    details: Dict[str, Any]

class BotRunner:
    """
    Enhanced bot runner with parallel execution, performance baselines,
    regression testing, and resource monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.agent_api = None
        self.initialized = False
        self.baselines_dir = Path("tests/performance_test_suite/baselines")
        self.baselines_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.resource_metrics: List[ResourceMetrics] = []
        self.regression_results: List[RegressionTestResult] = []
        
        # Resource monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("BotRunner initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enhanced testing."""
        return {
            "parallel_execution": {
                "enabled": True,
                "max_concurrent": 5,
                "rate_limit_per_second": 10
            },
            "performance_baselines": {
                "enabled": True,
                "baseline_threshold": 0.15,  # 15% degradation threshold
                "auto_update_baselines": False
            },
            "resource_monitoring": {
                "enabled": True,
                "monitoring_interval": 1.0,  # seconds
                "track_disk_io": True,
                "track_network": True
            },
            "regression_testing": {
                "enabled": True,
                "critical_paths": [
                    "server_analysis",
                    "semantic_search", 
                    "user_analysis",
                    "digest_generation"
                ]
            }
        }

    async def initialize(self):
        """Initialize the enhanced bot runner."""
        if self.initialized:
            return
        
        try:
            config = get_modernized_config()
            self.agent_api = AgentAPI(config)
            await self.agent_api.initialize()
            self.initialized = True
            logger.info("BotRunner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BotRunner: {e}")
            raise

    async def run_parallel_queries(
        self, 
        queries: List[Any], 
        max_concurrent: Optional[int] = None,
        rate_limit: Optional[int] = None
    ) -> List[Any]:
        """
        Execute queries in parallel with rate limiting and progress tracking.
        
        Args:
            queries: List of test queries to execute
            max_concurrent: Maximum concurrent queries (default from config)
            rate_limit: Queries per second limit (default from config)
            
        Returns:
            List of BotResponse objects
        """
        if not self.initialized:
            await self.initialize()
        
        max_concurrent = max_concurrent or self.config["parallel_execution"]["max_concurrent"]
        rate_limit = rate_limit or self.config["parallel_execution"]["rate_limit_per_second"]
        
        print_info("🚀 Starting parallel query execution...")
        print_info(f"   Total queries: {len(queries)}")
        print_info(f"   Max concurrent: {max_concurrent}")
        print_info(f"   Rate limit: {rate_limit}/sec")
        
        # Start resource monitoring
        if self.config["resource_monitoring"]["enabled"]:
            await self._start_resource_monitoring()
        
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create rate limiter
        rate_limiter = asyncio.Semaphore(rate_limit)
        
        # Execute queries in parallel
        tasks = []
        for i, query in enumerate(queries):
            task = self._execute_query_with_limits(
                query, i + 1, len(queries), semaphore, rate_limiter
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop resource monitoring
        if self.config["resource_monitoring"]["enabled"]:
            await self._stop_resource_monitoring()
        
        # Process results
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Handle exceptions
                processed_responses.append(self._create_error_response(
                    queries[i], f"Execution error: {response}"
                ))
            else:
                processed_responses.append(response)
        
        execution_time = time.time() - start_time
        
        # Generate performance report
        await self._generate_performance_report(processed_responses, execution_time)
        
        return processed_responses

    async def _execute_query_with_limits(
        self, 
        query: Any, 
        query_num: int, 
        total_queries: int,
        semaphore: asyncio.Semaphore,
        rate_limiter: asyncio.Semaphore
    ) -> Any:
        """Execute a single query with concurrency and rate limiting."""
        async with semaphore:
            async with rate_limiter:
                return await self._execute_single_query(query, query_num, total_queries)

    async def _execute_single_query(self, query: Any, query_num: int, total_queries: int) -> Any:
        """Execute a single query with timing and error handling."""
        start_time = time.time()
        
        try:
            # Get context
            context = await self._get_real_context_from_database()
            
            # Execute query
            result = await self.agent_api.query(
                query=query.query,
                user_id="test_user",
                context=context
            )
            
            response_time = time.time() - start_time
            
            # Create response object
            from tests.performance_test_suite.bot_runner import BotResponse
            
            return BotResponse(
                query_id=query.id,
                query=query.query,
                response=result.get("response", "No response"),
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                success=result.get("metadata", {}).get("success", True),
                metadata=result.get("metadata", {}),
                agent_used=result.get("metadata", {}).get("agent_used", "unknown"),
                routing_info=result.get("metadata", {}).get("routing_result"),
                validation_passed=result.get("metadata", {}).get("validation_passes", True)
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Query {query_num} failed: {e}")
            
            from tests.performance_test_suite.bot_runner import BotResponse
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

    def _create_error_response(self, query: Any, error_message: str) -> Any:
        """Create an error response object."""
        from tests.performance_test_suite.bot_runner import BotResponse
        
        return BotResponse(
            query_id=query.id,
            query=query.query,
            response=error_message,
            response_time=0.0,
            timestamp=datetime.now().isoformat(),
            success=False,
            metadata={"error": error_message},
            agent_used="error"
        )

    async def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring_active = True
        self.resource_metrics = []
        
        # Start monitoring in background
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")

    async def _stop_resource_monitoring(self):
        """Stop monitoring system resources."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        process = psutil.Process()
        interval = self.config["resource_monitoring"]["monitoring_interval"]
        
        while self.monitoring_active:
            try:
                # CPU and memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Disk I/O
                disk_io = process.io_counters()
                disk_io_read_mb = disk_io.read_bytes / 1024 / 1024
                disk_io_write_mb = disk_io.write_bytes / 1024 / 1024
                
                # Network I/O
                network_io = psutil.net_io_counters()
                network_sent_mb = network_io.bytes_sent / 1024 / 1024
                network_recv_mb = network_io.bytes_recv / 1024 / 1024
                
                metric = ResourceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    disk_io_read_mb=disk_io_read_mb,
                    disk_io_write_mb=disk_io_write_mb,
                    network_sent_mb=network_sent_mb,
                    network_recv_mb=network_recv_mb
                )
                
                self.resource_metrics.append(metric)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)

    async def create_performance_baseline(
        self, 
        name: str, 
        responses: List[Any]
    ) -> PerformanceBaseline:
        """Create a performance baseline from test results."""
        if not responses:
            raise ValueError("No responses provided for baseline creation")
        
        # Calculate performance metrics
        response_times = [r.response_time for r in responses if r.success]
        success_rate = sum(1 for r in responses if r.success) / len(responses)
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = p95_response_time = p99_response_time = 0.0
        
        # Calculate resource metrics
        if self.resource_metrics:
            max_memory_mb = max(m.memory_mb for m in self.resource_metrics)
            max_cpu_percent = max(m.cpu_percent for m in self.resource_metrics)
        else:
            max_memory_mb = max_cpu_percent = 0.0
        
        # Analyze agent usage
        agent_usage = {}
        for response in responses:
            agent = response.agent_used or "unknown"
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        baseline = PerformanceBaseline(
            name=name,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            success_rate=success_rate,
            max_memory_mb=max_memory_mb,
            max_cpu_percent=max_cpu_percent,
            timestamp=datetime.now().isoformat(),
            query_count=len(responses),
            agent_usage=agent_usage
        )
        
        # Save baseline
        baseline_file = self.baselines_dir / f"{name}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2, default=str)
        
        logger.info(f"Performance baseline '{name}' created and saved")
        return baseline

    async def run_regression_tests(
        self, 
        responses: List[Any], 
        baseline_name: str
    ) -> List[RegressionTestResult]:
        """Run regression tests against a performance baseline."""
        baseline_file = self.baselines_dir / f"{baseline_name}_baseline.json"
        
        if not baseline_file.exists():
            logger.warning(f"Baseline '{baseline_name}' not found, skipping regression tests")
            return []
        
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        baseline = PerformanceBaseline(**baseline_data)
        
        # Calculate current metrics
        response_times = [r.response_time for r in responses if r.success]
        success_rate = sum(1 for r in responses if r.success) / len(responses)
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]
            p99_response_time = statistics.quantiles(response_times, n=100)[98]
        else:
            avg_response_time = p95_response_time = p99_response_time = 0.0
        
        # Calculate resource metrics
        if self.resource_metrics:
            max_memory_mb = max(m.memory_mb for m in self.resource_metrics)
            max_cpu_percent = max(m.cpu_percent for m in self.resource_metrics)
        else:
            max_memory_mb = max_cpu_percent = 0.0
        
        # Calculate performance changes
        threshold = self.config["performance_baselines"]["baseline_threshold"]
        
        performance_degradation = (
            (avg_response_time - baseline.avg_response_time) / baseline.avg_response_time
            if baseline.avg_response_time > 0 else 0
        )
        
        memory_increase = (
            (max_memory_mb - baseline.max_memory_mb) / baseline.max_memory_mb
            if baseline.max_memory_mb > 0 else 0
        )
        
        cpu_increase = (
            (max_cpu_percent - baseline.max_cpu_percent) / baseline.max_cpu_percent
            if baseline.max_cpu_percent > 0 else 0
        )
        
        # Determine if tests passed
        passed = (
            performance_degradation <= threshold and
            memory_increase <= threshold and
            cpu_increase <= threshold and
            success_rate >= baseline.success_rate * 0.95  # Allow 5% success rate degradation
        )
        
        result = RegressionTestResult(
            test_name=f"regression_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            baseline_name=baseline_name,
            passed=passed,
            performance_degradation=performance_degradation,
            memory_increase=memory_increase,
            cpu_increase=cpu_increase,
            details={
                "current_avg_response_time": avg_response_time,
                "baseline_avg_response_time": baseline.avg_response_time,
                "current_success_rate": success_rate,
                "baseline_success_rate": baseline.success_rate,
                "current_max_memory_mb": max_memory_mb,
                "baseline_max_memory_mb": baseline.max_memory_mb,
                "current_max_cpu_percent": max_cpu_percent,
                "baseline_max_cpu_percent": baseline.max_cpu_percent
            }
        )
        
        self.regression_results.append(result)
        
        # Log results
        if passed:
            logger.info(f"✅ Regression test passed against baseline '{baseline_name}'")
        else:
            logger.warning(f"⚠️ Regression test failed against baseline '{baseline_name}'")
            logger.warning(f"   Performance degradation: {performance_degradation:.2%}")
            logger.warning(f"   Memory increase: {memory_increase:.2%}")
            logger.warning(f"   CPU increase: {cpu_increase:.2%}")
        
        return [result]

    async def _generate_performance_report(
        self, 
        responses: List[Any], 
        execution_time: float
    ):
        """Generate comprehensive performance report."""
        if not responses:
            return
        
        # Calculate metrics
        successful_responses = [r for r in responses if r.success]
        failed_responses = [r for r in responses if r.failed]
        
        success_rate = len(successful_responses) / len(responses)
        avg_response_time = statistics.mean([r.response_time for r in successful_responses]) if successful_responses else 0
        
        # Resource metrics
        max_memory_mb = max(m.memory_mb for m in self.resource_metrics) if self.resource_metrics else 0
        max_cpu_percent = max(m.cpu_percent for m in self.resource_metrics) if self.resource_metrics else 0
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "total_queries": len(responses),
            "successful_queries": len(successful_responses),
            "failed_queries": len(failed_responses),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "max_memory_mb": max_memory_mb,
            "max_cpu_percent": max_cpu_percent,
            "resource_metrics_count": len(self.resource_metrics),
            "regression_results": [asdict(r) for r in self.regression_results]
        }
        
        # Save report
        reports_dir = Path("tests/performance_test_suite/reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"performance_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {report_file}")
        
        # Print summary
        print_success("📊 Performance Report Summary")
        print_info(f"   Execution time: {execution_time:.2f}s")
        print_info(f"   Success rate: {success_rate:.1%}")
        print_info(f"   Avg response time: {avg_response_time:.2f}s")
        print_info(f"   Max memory: {max_memory_mb:.1f}MB")
        print_info(f"   Max CPU: {max_cpu_percent:.1f}%")

    async def _get_real_context_from_database(self) -> Dict[str, Any]:
        """Get real context from the database for authentic testing."""
        try:
            db_path = Path("data/discord_messages.db")
            if not db_path.exists():
                return self._get_fallback_context()
            
            conn = sqlite3.connect(db_path)
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

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.agent_api and hasattr(self.agent_api, 'cleanup'):
                await self.agent_api.cleanup()
            
            if self.monitoring_active:
                await self._stop_resource_monitoring()
            
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