"""
CI/CD Integration for Performance Testing

Automated testing integration for deployment pipelines with:
- Critical path regression testing
- Performance baseline validation
- Automated reporting
- Deployment gate validation
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.performance_test_suite.bot_runner_v2 import BotRunner
from tests.performance_test_suite.query_generator import QueryGenerator
from tests.performance_test_suite.synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

class CICDPerformanceTester:
    """
    CI/CD performance testing integration.
    
    Designed for automated testing in deployment pipelines with:
    - Fast execution for CI/CD constraints
    - Critical path validation
    - Performance regression detection
    - Automated pass/fail decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        self.deployment_gate_passed = False
        
        # Initialize components
        self.bot_runner = BotRunner(self.config.get("bot_config"))
        self.query_generator = QueryGenerator()
        self.synthetic_generator = SyntheticDataGenerator()
        
        logger.info("CICDPerformanceTester initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for CI/CD testing."""
        return {
            "timeout_seconds": 300,  # 5 minutes max
            "max_queries": 10,  # Limited for CI/CD
            "critical_paths": [
                "server_analysis",
                "semantic_search",
                "user_analysis"
            ],
            "performance_thresholds": {
                "max_avg_response_time": 5.0,  # seconds
                "min_success_rate": 0.8,  # 80%
                "max_memory_mb": 1024,  # 1GB
                "max_cpu_percent": 80.0
            },
            "regression_threshold": 0.2,  # 20% degradation allowed
            "bot_config": {
                "parallel_execution": {
                    "enabled": True,
                    "max_concurrent": 3,
                    "rate_limit_per_second": 5
                },
                "performance_baselines": {
                    "enabled": True,
                    "baseline_threshold": 0.2,
                    "auto_update_baselines": False
                },
                "resource_monitoring": {
                    "enabled": True,
                    "monitoring_interval": 1.0
                }
            }
        }

    async def run_cicd_tests(self) -> Dict[str, Any]:
        """
        Run CI/CD performance tests.
        
        Returns:
            Test results with deployment gate decision
        """
        start_time = time.time()
        
        print("🚀 CI/CD Performance Testing")
        print("=" * 50)
        print(f"⏱️ Timeout: {self.config['timeout_seconds']}s")
        print(f"📊 Max queries: {self.config['max_queries']}")
        print(f"🎯 Critical paths: {len(self.config['critical_paths'])}")
        
        try:
            # Phase 1: Critical Path Testing
            critical_results = await self._run_critical_path_tests()
            
            # Phase 2: Performance Baseline Validation
            baseline_results = await self._run_baseline_validation()
            
            # Phase 3: Resource Usage Validation
            resource_results = await self._run_resource_validation()
            
            # Phase 4: Edge Case Testing
            edge_case_results = await self._run_edge_case_tests()
            
            # Phase 5: Deployment Gate Decision
            gate_decision = self._make_deployment_gate_decision(
                critical_results,
                baseline_results,
                resource_results,
                edge_case_results
            )
            
            # Generate results
            execution_time = time.time() - start_time
            results = {
                "status": "completed",
                "deployment_gate_passed": gate_decision["passed"],
                "execution_time": execution_time,
                "critical_path_tests": critical_results,
                "baseline_validation": baseline_results,
                "resource_validation": resource_results,
                "edge_case_tests": edge_case_results,
                "gate_decision": gate_decision,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results = results
            self.deployment_gate_passed = gate_decision["passed"]
            
            # Print results
            self._print_cicd_results(results)
            
            return results
            
        except asyncio.TimeoutError:
            print("⏰ CI/CD tests timed out!")
            return {
                "status": "timeout",
                "deployment_gate_passed": False,
                "error": "Tests exceeded timeout limit"
            }
        except Exception as e:
            print(f"❌ CI/CD tests failed: {e}")
            return {
                "status": "failed",
                "deployment_gate_passed": False,
                "error": str(e)
            }

    async def _run_critical_path_tests(self) -> Dict[str, Any]:
        """Run tests for critical system paths."""
        print("\n🎯 Phase 1: Critical Path Testing")
        
        try:
            # Generate critical path queries
            critical_queries = []
            for path in self.config["critical_paths"]:
                if path == "server_analysis":
                    critical_queries.append(self._create_test_query(
                        "What are the most active channels in the server?",
                        "server_analysis"
                    ))
                elif path == "semantic_search":
                    critical_queries.append(self._create_test_query(
                        "Find messages about AI and machine learning",
                        "semantic_search"
                    ))
                elif path == "user_analysis":
                    critical_queries.append(self._create_test_query(
                        "Show me user engagement patterns",
                        "user_analysis"
                    ))
            
            # Execute critical path queries
            responses = await self.bot_runner.run_parallel_queries(
                critical_queries,
                max_concurrent=2,
                rate_limit=3
            )
            
            # Analyze results
            successful = sum(1 for r in responses if r.success)
            total = len(responses)
            success_rate = successful / total if total > 0 else 0
            
            avg_response_time = sum(r.response_time for r in responses if r.success) / successful if successful > 0 else 0
            
            results = {
                "total_queries": total,
                "successful_queries": successful,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "passed": success_rate >= 0.8 and avg_response_time <= 10.0,
                "responses": [self._serialize_response(r) for r in responses]
            }
            
            print(f"   ✅ Critical paths: {successful}/{total} passed")
            print(f"   📊 Success rate: {success_rate:.1%}")
            print(f"   ⚡ Avg response time: {avg_response_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Critical path tests failed: {e}")
            return {
                "error": str(e),
                "passed": False
            }

    async def _run_baseline_validation(self) -> Dict[str, Any]:
        """Run performance baseline validation."""
        print("\n📊 Phase 2: Performance Baseline Validation")
        
        try:
            # Generate baseline test queries
            baseline_queries = [
                self._create_test_query("What are the most active channels?", "baseline"),
                self._create_test_query("Show me recent activity", "baseline"),
                self._create_test_query("Find messages about AI", "baseline")
            ]
            
            # Execute baseline queries
            responses = await self.bot_runner.run_parallel_queries(
                baseline_queries,
                max_concurrent=2,
                rate_limit=3
            )
            
            # Check against baseline
            baseline_name = "production_baseline"
            regression_results = await self.bot_runner.run_regression_tests(
                responses, baseline_name
            )
            
            if regression_results:
                regression_passed = regression_results[0].passed
                performance_degradation = regression_results[0].performance_degradation
            else:
                # No baseline exists, create one
                await self.bot_runner.create_performance_baseline(baseline_name, responses)
                regression_passed = True
                performance_degradation = 0.0
            
            results = {
                "baseline_exists": len(regression_results) > 0,
                "regression_passed": regression_passed,
                "performance_degradation": performance_degradation,
                "passed": regression_passed,
                "regression_results": [self._serialize_regression_result(r) for r in regression_results]
            }
            
            print(f"   ✅ Baseline validation: {'PASSED' if regression_passed else 'FAILED'}")
            if regression_results:
                print(f"   📉 Performance degradation: {performance_degradation:.1%}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Baseline validation failed: {e}")
            return {
                "error": str(e),
                "passed": False
            }

    async def _run_resource_validation(self) -> Dict[str, Any]:
        """Run resource usage validation."""
        print("\n💾 Phase 3: Resource Usage Validation")
        
        try:
            # Generate resource-intensive queries
            resource_queries = [
                self._create_test_query("Generate a comprehensive weekly digest", "resource"),
                self._create_test_query("Analyze all user engagement patterns", "resource")
            ]
            
            # Execute with resource monitoring
            responses = await self.bot_runner.run_parallel_queries(
                resource_queries,
                max_concurrent=1,  # Sequential for resource testing
                rate_limit=2
            )
            
            # Analyze resource usage
            if self.bot_runner.resource_metrics:
                max_memory_mb = max(m.memory_mb for m in self.bot_runner.resource_metrics)
                max_cpu_percent = max(m.cpu_percent for m in self.bot_runner.resource_metrics)
            else:
                max_memory_mb = 0
                max_cpu_percent = 0
            
            # Check against thresholds
            memory_passed = max_memory_mb <= self.config["performance_thresholds"]["max_memory_mb"]
            cpu_passed = max_cpu_percent <= self.config["performance_thresholds"]["max_cpu_percent"]
            
            results = {
                "max_memory_mb": max_memory_mb,
                "max_cpu_percent": max_cpu_percent,
                "memory_threshold": self.config["performance_thresholds"]["max_memory_mb"],
                "cpu_threshold": self.config["performance_thresholds"]["max_cpu_percent"],
                "memory_passed": memory_passed,
                "cpu_passed": cpu_passed,
                "passed": memory_passed and cpu_passed
            }
            
            print(f"   💾 Max memory: {max_memory_mb:.1f}MB (threshold: {self.config['performance_thresholds']['max_memory_mb']}MB)")
            print(f"   🔥 Max CPU: {max_cpu_percent:.1f}% (threshold: {self.config['performance_thresholds']['max_cpu_percent']}%)")
            print(f"   ✅ Resource validation: {'PASSED' if results['passed'] else 'FAILED'}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Resource validation failed: {e}")
            return {
                "error": str(e),
                "passed": False
            }

    async def _run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case tests."""
        print("\n🧪 Phase 4: Edge Case Testing")
        
        try:
            # Generate edge case scenarios
            edge_scenarios = self.synthetic_generator.generate_edge_case_scenarios()
            
            # Take first few queries from each scenario for CI/CD
            edge_queries = []
            for scenario in edge_scenarios[:2]:  # Limit for CI/CD
                edge_queries.extend(scenario.queries[:2])  # 2 queries per scenario
            
            # Execute edge case queries
            responses = await self.bot_runner.run_parallel_queries(
                edge_queries,
                max_concurrent=2,
                rate_limit=3
            )
            
            # Analyze results
            successful = sum(1 for r in responses if r.success)
            total = len(responses)
            success_rate = successful / total if total > 0 else 0
            
            # Edge cases should handle gracefully (not necessarily succeed)
            graceful_handling = sum(1 for r in responses if r.success or "error" in r.response.lower()) / total if total > 0 else 0
            
            results = {
                "total_queries": total,
                "successful_queries": successful,
                "success_rate": success_rate,
                "graceful_handling_rate": graceful_handling,
                "passed": graceful_handling >= 0.8,  # 80% should handle gracefully
                "responses": [self._serialize_response(r) for r in responses]
            }
            
            print(f"   🧪 Edge cases: {successful}/{total} succeeded")
            print(f"   🛡️ Graceful handling: {graceful_handling:.1%}")
            print(f"   ✅ Edge case testing: {'PASSED' if results['passed'] else 'FAILED'}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Edge case testing failed: {e}")
            return {
                "error": str(e),
                "passed": False
            }

    def _make_deployment_gate_decision(
        self,
        critical_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        resource_results: Dict[str, Any],
        edge_case_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make deployment gate decision based on all test results."""
        print("\n🚪 Phase 5: Deployment Gate Decision")
        
        # Check each component
        critical_passed = critical_results.get("passed", False)
        baseline_passed = baseline_results.get("passed", False)
        resource_passed = resource_results.get("passed", False)
        edge_case_passed = edge_case_results.get("passed", False)
        
        # All components must pass
        overall_passed = all([
            critical_passed,
            baseline_passed,
            resource_passed,
            edge_case_passed
        ])
        
        decision = {
            "passed": overall_passed,
            "critical_paths": critical_passed,
            "baseline_validation": baseline_passed,
            "resource_validation": resource_passed,
            "edge_case_handling": edge_case_passed,
            "reason": self._get_failure_reason(
                critical_passed, baseline_passed, resource_passed, edge_case_passed
            )
        }
        
        print(f"   🎯 Critical paths: {'✅ PASSED' if critical_passed else '❌ FAILED'}")
        print(f"   📊 Baseline validation: {'✅ PASSED' if baseline_passed else '❌ FAILED'}")
        print(f"   💾 Resource validation: {'✅ PASSED' if resource_passed else '❌ FAILED'}")
        print(f"   🧪 Edge case handling: {'✅ PASSED' if edge_case_passed else '❌ FAILED'}")
        print(f"   🚪 Deployment gate: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        if not overall_passed:
            print(f"   📝 Reason: {decision['reason']}")
        
        return decision

    def _get_failure_reason(
        self,
        critical_passed: bool,
        baseline_passed: bool,
        resource_passed: bool,
        edge_case_passed: bool
    ) -> str:
        """Get human-readable failure reason."""
        failures = []
        
        if not critical_passed:
            failures.append("Critical path tests failed")
        if not baseline_passed:
            failures.append("Performance regression detected")
        if not resource_passed:
            failures.append("Resource usage exceeded thresholds")
        if not edge_case_passed:
            failures.append("Edge case handling inadequate")
        
        return "; ".join(failures) if failures else "Unknown failure"

    def _create_test_query(self, query: str, category: str) -> Any:
        """Create a test query object."""
        from tests.performance_test_suite.query_generator import TestQuery
        
        return TestQuery(
            id=f"cicd_{category}_{int(time.time())}",
            query=query,
            category=category,
            subcategory="cicd_test",
            complexity="moderate",
            expected_response_structure={
                "cicd_test": True,
                "timeout": 30.0
            }
        )

    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """Serialize response for JSON output."""
        return {
            "query_id": response.query_id,
            "query": response.query,
            "success": response.success,
            "response_time": response.response_time,
            "agent_used": response.agent_used,
            "error": response.metadata.get("error") if not response.success else None
        }

    def _serialize_regression_result(self, result: Any) -> Dict[str, Any]:
        """Serialize regression result for JSON output."""
        return {
            "test_name": result.test_name,
            "baseline_name": result.baseline_name,
            "passed": result.passed,
            "performance_degradation": result.performance_degradation,
            "memory_increase": result.memory_increase,
            "cpu_increase": result.cpu_increase
        }

    def _print_cicd_results(self, results: Dict[str, Any]):
        """Print CI/CD test results."""
        print("\n" + "=" * 50)
        print("📊 CI/CD Test Results")
        print("=" * 50)
        
        print(f"⏱️ Execution time: {results.get('execution_time', 0):.2f}s")
        print(f"🚪 Deployment gate: {'✅ PASSED' if results.get('deployment_gate_passed') else '❌ FAILED'}")
        
        if results.get("status") == "completed":
            print(f"\n📋 Component Results:")
            
            critical = results.get("critical_path_tests", {})
            print(f"   🎯 Critical paths: {critical.get('successful_queries', 0)}/{critical.get('total_queries', 0)} passed")
            
            baseline = results.get("baseline_validation", {})
            print(f"   📊 Baseline validation: {'✅ PASSED' if baseline.get('passed') else '❌ FAILED'}")
            
            resource = results.get("resource_validation", {})
            print(f"   💾 Resource validation: {'✅ PASSED' if resource.get('passed') else '❌ FAILED'}")
            
            edge_case = results.get("edge_case_tests", {})
            print(f"   🧪 Edge case handling: {edge_case.get('successful_queries', 0)}/{edge_case.get('total_queries', 0)} passed")
        
        # Save results for CI/CD systems
        self._save_cicd_results(results)

    def _save_cicd_results(self, results: Dict[str, Any]):
        """Save results for CI/CD systems."""
        try:
            # Save to file
            results_dir = Path("tests/performance_test_suite/cicd_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"cicd_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Also save latest results
            latest_file = results_dir / "latest_results.json"
            with open(latest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📄 Results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self.bot_runner, 'cleanup'):
                await self.bot_runner.cleanup()
            logger.info("CICDPerformanceTester cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point for CI/CD testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI/CD Performance Testing")
    parser.add_argument('--timeout', type=int, default=300, help='Test timeout in seconds')
    parser.add_argument('--max-queries', type=int, default=10, help='Maximum queries to run')
    parser.add_argument('--baseline', type=str, default='production_baseline', help='Baseline name')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "timeout_seconds": args.timeout,
        "max_queries": args.max_queries,
        "performance_baselines": {
            "baseline_name": args.baseline
        }
    }
    
    # Run CI/CD tests
    tester = CICDPerformanceTester(config)
    
    try:
        results = await tester.run_cicd_tests()
        
        # Exit with appropriate code
        if results.get("deployment_gate_passed"):
            print("\n🎉 CI/CD tests passed - deployment can proceed!")
            return 0
        else:
            print("\n❌ CI/CD tests failed - deployment blocked!")
            return 1
            
    except Exception as e:
        print(f"\n❌ CI/CD tests failed with error: {e}")
        return 1
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 