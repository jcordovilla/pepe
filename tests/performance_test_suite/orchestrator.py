"""
Enhanced Performance Test Orchestrator

Advanced orchestrator that integrates parallel execution, performance baselines,
regression testing, synthetic data generation, and resource monitoring.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

# Import components
from tests.performance_test_suite.bot_runner_v2 import BotRunner
from tests.performance_test_suite.synthetic_data_generator import SyntheticDataGenerator
from tests.performance_test_suite.content_analyzer import ContentAnalyzer
from tests.performance_test_suite.query_generator import QueryGenerator
from tests.performance_test_suite.evaluator import ResponseEvaluator
from tests.performance_test_suite.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# Color utility
class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_phase(msg):
    print(f"{Color.HEADER}{Color.BOLD}\n=== {msg} ==={Color.ENDC}")
    sys.stdout.flush()

def print_success(msg):
    print(f"{Color.OKGREEN}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_warning(msg):
    print(f"{Color.WARNING}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_error(msg):
    print(f"{Color.FAIL}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_info(msg):
    print(f"{Color.OKBLUE}{msg}{Color.ENDC}")
    sys.stdout.flush()

def print_progress(phase: str, current: int, total: int, description: str = ""):
    """Print progress with timestamp and percentage."""
    percentage = (current / total) * 100 if total > 0 else 0
    timestamp = datetime.now().strftime("%H:%M:%S")
    progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    print(f"{Color.OKCYAN}[{timestamp}] {phase}: [{progress_bar}] {current}/{total} ({percentage:.1f}%) {description}{Color.ENDC}")
    sys.stdout.flush()


class PerformanceTestOrchestrator:
    """
    Enhanced orchestrator for comprehensive performance testing.
    
    Features:
    - Parallel query execution
    - Performance baselines and regression testing
    - Synthetic data generation for edge cases and stress testing
    - Resource monitoring
    - Automated critical path testing
    - Enhanced reporting with trend analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        self.start_time = None
        self.phase_times = {}
        
        # Initialize enhanced components
        self.content_analyzer = ContentAnalyzer(self.config.get("database_path"))
        self.query_generator = QueryGenerator()
        self.bot_runner = BotRunner(self.config.get("bot_config"))
        self.synthetic_generator = SyntheticDataGenerator()
        self.evaluator = ResponseEvaluator()
        self.report_generator = ReportGenerator()
        
        logger.info("PerformanceTestOrchestrator initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enhanced testing."""
        return {
            "database_path": "data/discord_messages.db",
            "bot_api_endpoint": None,  # Use local AgentAPI
            "output_directory": "tests/performance_test_suite/data",
            "sample_percentage": 0.15,
            "query_count": 50,
            "enable_error_scenarios": True,
            "save_intermediate_results": True,
            
            # Enhanced features
            "parallel_execution": {
                "enabled": True,
                "max_concurrent": 5,
                "rate_limit_per_second": 10
            },
            "performance_baselines": {
                "enabled": True,
                "baseline_threshold": 0.15,
                "auto_update_baselines": False,
                "baseline_name": "production_baseline"
            },
            "regression_testing": {
                "enabled": True,
                "critical_paths": [
                    "server_analysis",
                    "semantic_search",
                    "user_analysis",
                    "digest_generation"
                ]
            },
            "synthetic_testing": {
                "enabled": True,
                "edge_cases": True,
                "stress_scenarios": True,
                "load_testing": True
            },
            "resource_monitoring": {
                "enabled": True,
                "monitoring_interval": 1.0
            },
            "enhanced_bot_config": {
                "parallel_execution": {
                    "enabled": True,
                    "max_concurrent": 5,
                    "rate_limit_per_second": 10
                },
                "performance_baselines": {
                    "enabled": True,
                    "baseline_threshold": 0.15,
                    "auto_update_baselines": False
                },
                "resource_monitoring": {
                    "enabled": True,
                    "monitoring_interval": 1.0,
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
        }

    async def run_enhanced_performance_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive enhanced performance tests.
        
        Returns:
            Comprehensive test results with performance analysis
        """
        self.start_time = time.time()
        
        print_phase("🚀 Enhanced Performance Test Suite")
        print_info(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"⚙️ Parallel execution: {'Enabled' if self.config['parallel_execution']['enabled'] else 'Disabled'}")
        print_info(f"📊 Performance baselines: {'Enabled' if self.config['performance_baselines']['enabled'] else 'Disabled'}")
        print_info(f"🔄 Regression testing: {'Enabled' if self.config['regression_testing']['enabled'] else 'Disabled'}")
        print_info(f"🧪 Synthetic testing: {'Enabled' if self.config['synthetic_testing']['enabled'] else 'Disabled'}")
        
        try:
            # Phase 1: Content Analysis
            content_analysis = await self._run_content_analysis()
            
            # Phase 2: Standard Query Generation
            standard_queries = await self._run_standard_query_generation(content_analysis)
            
            # Phase 3: Synthetic Data Generation
            synthetic_scenarios = await self._run_synthetic_data_generation()
            
            # Phase 4: Parallel Query Execution
            standard_responses = await self._run_parallel_query_execution(standard_queries)
            
            # Phase 5: Synthetic Scenario Execution
            synthetic_responses = await self._run_synthetic_scenario_execution(synthetic_scenarios)
            
            # Phase 6: Performance Baseline Creation/Testing
            baseline_results = await self._run_performance_baseline_analysis(standard_responses)
            
            # Phase 7: Regression Testing
            regression_results = await self._run_regression_testing(standard_responses)
            
            # Phase 8: Response Evaluation
            evaluations = await self._run_enhanced_response_evaluation(
                standard_queries + synthetic_scenarios,
                standard_responses + synthetic_responses
            )
            
            # Phase 9: Enhanced Report Generation
            report = await self._run_enhanced_report_generation(
                standard_queries,
                standard_responses,
                synthetic_scenarios,
                synthetic_responses,
                evaluations,
                baseline_results,
                regression_results
            )
            
            # Generate final summary
            summary = self._generate_final_summary(
                standard_responses,
                synthetic_responses,
                baseline_results,
                regression_results
            )
            
            return {
                "status": "completed",
                "summary": summary,
                "report": report,
                "baseline_results": baseline_results,
                "regression_results": regression_results,
                "execution_time": time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Enhanced performance tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - self.start_time
            }
        finally:
            await self.cleanup()

    async def _run_content_analysis(self) -> Dict[str, Any]:
        """Run content analysis phase."""
        print_phase("📊 Phase 1: Content Analysis")
        
        start_time = time.time()
        
        try:
            self.content_analyzer.preflight_schema_check()
            content_analysis = await self.content_analyzer.analyze_content()
            
            duration = time.time() - start_time
            self.phase_times["content_analysis"] = duration
            
            print_success(f"✅ Content analysis completed in {duration:.2f}s")
            print_info(f"   - Total messages: {content_analysis.get('total_messages', 0):,}")
            print_info(f"   - Active channels: {content_analysis.get('active_channels', 0)}")
            print_info(f"   - Active users: {content_analysis.get('active_users', 0)}")
            
            return content_analysis
            
        except Exception as e:
            print_error(f"❌ Content analysis failed: {e}")
            raise

    async def _run_standard_query_generation(self, content_analysis: Dict[str, Any]) -> List[Any]:
        """Run standard query generation phase."""
        print_phase("🎯 Phase 2: Standard Query Generation")
        
        start_time = time.time()
        
        try:
            self.query_generator.content_analysis = content_analysis
            queries = self.query_generator.generate_test_queries()
            
            duration = time.time() - start_time
            self.phase_times["query_generation"] = duration
            
            print_success(f"✅ Generated {len(queries)} standard queries in {duration:.2f}s")
            
            # Save queries
            if self.config["save_intermediate_results"]:
                self.query_generator.save_queries()
            
            return queries
            
        except Exception as e:
            print_error(f"❌ Query generation failed: {e}")
            raise

    async def _run_synthetic_data_generation(self) -> List[Any]:
        """Run synthetic data generation phase."""
        if not self.config["synthetic_testing"]["enabled"]:
            print_info("⏭️ Skipping synthetic data generation (disabled)")
            return []
        
        print_phase("🧪 Phase 3: Synthetic Data Generation")
        
        start_time = time.time()
        
        try:
            all_scenarios = []
            
            # Generate edge case scenarios
            if self.config["synthetic_testing"]["edge_cases"]:
                print_info("   Generating edge case scenarios...")
                edge_scenarios = self.synthetic_generator.generate_edge_case_scenarios()
                all_scenarios.extend(edge_scenarios)
                print_success(f"   ✅ Generated {len(edge_scenarios)} edge case scenarios")
            
            # Generate stress scenarios
            if self.config["synthetic_testing"]["stress_scenarios"]:
                print_info("   Generating stress scenarios...")
                stress_scenarios = self.synthetic_generator.generate_stress_scenarios()
                all_scenarios.extend(stress_scenarios)
                print_success(f"   ✅ Generated {len(stress_scenarios)} stress scenarios")
            
            # Generate load test scenarios
            if self.config["synthetic_testing"]["load_testing"]:
                print_info("   Generating load test scenarios...")
                load_scenarios = self.synthetic_generator.generate_load_test_scenarios()
                all_scenarios.extend(load_scenarios)
                print_success(f"   ✅ Generated {len(load_scenarios)} load test scenarios")
            
            # Extract all queries from scenarios
            all_queries = []
            for scenario in all_scenarios:
                all_queries.extend(scenario.queries)
            
            duration = time.time() - start_time
            self.phase_times["synthetic_generation"] = duration
            
            print_success(f"✅ Generated {len(all_queries)} synthetic queries in {duration:.2f}s")
            
            # Save scenarios
            if self.config["save_intermediate_results"]:
                self.synthetic_generator.save_scenarios(all_scenarios)
            
            return all_queries
            
        except Exception as e:
            print_error(f"❌ Synthetic data generation failed: {e}")
            raise

    async def _run_parallel_query_execution(self, queries: List[Any]) -> List[Any]:
        """Run parallel query execution phase."""
        print_phase("⚡ Phase 4: Parallel Query Execution")
        
        start_time = time.time()
        
        try:
            if self.config["parallel_execution"]["enabled"]:
                print_info("🚀 Using parallel execution...")
                responses = await self.bot_runner.run_parallel_queries(
                    queries,
                    max_concurrent=self.config["parallel_execution"]["max_concurrent"],
                    rate_limit=self.config["parallel_execution"]["rate_limit_per_second"]
                )
            else:
                print_info("🐌 Using sequential execution...")
                responses = await self.bot_runner.run_queries_with_progress(queries)
            
            duration = time.time() - start_time
            self.phase_times["query_execution"] = duration
            
            print_success(f"✅ Executed {len(queries)} queries in {duration:.2f}s")
            print_info(f"   - Successful: {sum(1 for r in responses if r.success)}")
            print_info(f"   - Failed: {sum(1 for r in responses if not r.success)}")
            
            return responses
            
        except Exception as e:
            print_error(f"❌ Query execution failed: {e}")
            raise

    async def _run_synthetic_scenario_execution(self, synthetic_queries: List[Any]) -> List[Any]:
        """Run synthetic scenario execution phase."""
        if not synthetic_queries:
            print_info("⏭️ No synthetic queries to execute")
            return []
        
        print_phase("🧪 Phase 5: Synthetic Scenario Execution")
        
        start_time = time.time()
        
        try:
            # Execute synthetic queries with higher concurrency for stress testing
            responses = await self.bot_runner.run_parallel_queries(
                synthetic_queries,
                max_concurrent=10,  # Higher concurrency for stress testing
                rate_limit=20  # Higher rate limit for stress testing
            )
            
            duration = time.time() - start_time
            self.phase_times["synthetic_execution"] = duration
            
            print_success(f"✅ Executed {len(synthetic_queries)} synthetic queries in {duration:.2f}s")
            print_info(f"   - Successful: {sum(1 for r in responses if r.success)}")
            print_info(f"   - Failed: {sum(1 for r in responses if not r.success)}")
            
            return responses
            
        except Exception as e:
            print_error(f"❌ Synthetic scenario execution failed: {e}")
            raise

    async def _run_performance_baseline_analysis(self, responses: List[Any]) -> Dict[str, Any]:
        """Run performance baseline analysis phase."""
        if not self.config["performance_baselines"]["enabled"]:
            print_info("⏭️ Skipping performance baseline analysis (disabled)")
            return {}
        
        print_phase("📊 Phase 6: Performance Baseline Analysis")
        
        start_time = time.time()
        
        try:
            baseline_name = self.config["performance_baselines"]["baseline_name"]
            
            # Check if baseline exists
            baseline_file = Path(f"tests/performance_test_suite/baselines/{baseline_name}_baseline.json")
            
            if baseline_file.exists():
                print_info(f"📊 Running regression test against baseline: {baseline_name}")
                regression_results = await self.bot_runner.run_regression_tests(
                    responses, baseline_name
                )
                
                duration = time.time() - start_time
                self.phase_times["baseline_analysis"] = duration
                
                print_success(f"✅ Baseline analysis completed in {duration:.2f}s")
                
                return {
                    "baseline_exists": True,
                    "baseline_name": baseline_name,
                    "regression_results": regression_results
                }
            else:
                print_info(f"📊 Creating new performance baseline: {baseline_name}")
                baseline = await self.bot_runner.create_performance_baseline(
                    baseline_name, responses
                )
                
                duration = time.time() - start_time
                self.phase_times["baseline_analysis"] = duration
                
                print_success(f"✅ Baseline created in {duration:.2f}s")
                print_info(f"   - Avg response time: {baseline.avg_response_time:.2f}s")
                print_info(f"   - Success rate: {baseline.success_rate:.1%}")
                print_info(f"   - Max memory: {baseline.max_memory_mb:.1f}MB")
                
                return {
                    "baseline_exists": False,
                    "baseline_name": baseline_name,
                    "baseline": baseline
                }
                
        except Exception as e:
            print_error(f"❌ Performance baseline analysis failed: {e}")
            raise

    async def _run_regression_testing(self, responses: List[Any]) -> List[Any]:
        """Run regression testing phase."""
        if not self.config["regression_testing"]["enabled"]:
            print_info("⏭️ Skipping regression testing (disabled)")
            return []
        
        print_phase("🔄 Phase 7: Regression Testing")
        
        start_time = time.time()
        
        try:
            # Run critical path regression tests
            critical_paths = self.config["regression_testing"]["critical_paths"]
            regression_results = []
            
            for path in critical_paths:
                print_info(f"   Testing critical path: {path}")
                # Filter responses for this critical path
                path_responses = [r for r in responses if path in r.query.lower()]
                
                if path_responses:
                    path_results = await self.enhanced_bot_runner.run_regression_tests(
                        path_responses, f"{path}_baseline"
                    )
                    regression_results.extend(path_results)
            
            duration = time.time() - start_time
            self.phase_times["regression_testing"] = duration
            
            print_success(f"✅ Regression testing completed in {duration:.2f}s")
            print_info(f"   - Critical paths tested: {len(critical_paths)}")
            print_info(f"   - Regression results: {len(regression_results)}")
            
            return regression_results
            
        except Exception as e:
            print_error(f"❌ Regression testing failed: {e}")
            raise

    async def _run_enhanced_response_evaluation(
        self, 
        queries: List[Any], 
        responses: List[Any]
    ) -> List[Any]:
        """Run enhanced response evaluation phase."""
        print_phase("📋 Phase 8: Enhanced Response Evaluation")
        
        start_time = time.time()
        
        try:
            evaluations = self.evaluator.evaluate_responses(queries, responses)
            
            duration = time.time() - start_time
            self.phase_times["response_evaluation"] = duration
            
            print_success(f"✅ Evaluated {len(evaluations)} responses in {duration:.2f}s")
            
            # Calculate evaluation statistics
            avg_score = sum(e.overall_score for e in evaluations) / len(evaluations) if evaluations else 0
            print_info(f"   - Average score: {avg_score:.2f}")
            
            return evaluations
            
        except Exception as e:
            print_error(f"❌ Response evaluation failed: {e}")
            raise

    async def _run_enhanced_report_generation(
        self,
        standard_queries: List[Any],
        standard_responses: List[Any],
        synthetic_scenarios: List[Any],
        synthetic_responses: List[Any],
        evaluations: List[Any],
        baseline_results: Dict[str, Any],
        regression_results: List[Any]
    ) -> Dict[str, Any]:
        """Run enhanced report generation phase."""
        print_phase("📊 Phase 9: Enhanced Report Generation")
        
        start_time = time.time()
        
        try:
            # Generate comprehensive report
            report = await self.report_generator.generate_comprehensive_report(
                standard_queries,
                standard_responses,
                evaluations,
                baseline_results=baseline_results,
                regression_results=regression_results,
                synthetic_data={
                    "scenarios": synthetic_scenarios,
                    "responses": synthetic_responses
                },
                phase_times=self.phase_times
            )
            
            duration = time.time() - start_time
            self.phase_times["report_generation"] = duration
            
            print_success(f"✅ Report generation completed in {duration:.2f}s")
            
            return report
            
        except Exception as e:
            print_error(f"❌ Report generation failed: {e}")
            raise

    def _generate_final_summary(
        self,
        standard_responses: List[Any],
        synthetic_responses: List[Any],
        baseline_results: Dict[str, Any],
        regression_results: List[Any]
    ) -> Dict[str, Any]:
        """Generate final test summary."""
        total_responses = len(standard_responses) + len(synthetic_responses)
        successful_responses = sum(1 for r in standard_responses + synthetic_responses if r.success)
        
        # Calculate performance metrics
        response_times = [r.response_time for r in standard_responses + synthetic_responses if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Regression test results
        regression_passed = sum(1 for r in regression_results if r.passed) if regression_results else 0
        regression_total = len(regression_results) if regression_results else 0
        
        summary = {
            "total_execution_time": time.time() - self.start_time,
            "total_queries": total_responses,
            "successful_queries": successful_responses,
            "success_rate": successful_responses / total_responses if total_responses > 0 else 0,
            "avg_response_time": avg_response_time,
            "regression_tests_passed": regression_passed,
            "regression_tests_total": regression_total,
            "regression_success_rate": regression_passed / regression_total if regression_total > 0 else 0,
            "phase_times": self.phase_times,
            "baseline_analysis": baseline_results.get("baseline_exists", False),
            "synthetic_testing_enabled": self.config["synthetic_testing"]["enabled"]
        }
        
        return summary

    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self.bot_runner, 'cleanup'):
                await self.bot_runner.cleanup()
            logger.info("PerformanceTestOrchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point for enhanced performance testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Performance Test Suite")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--baselines', action='store_true', help='Enable performance baselines')
    parser.add_argument('--regression', action='store_true', help='Enable regression testing')
    parser.add_argument('--synthetic', action='store_true', help='Enable synthetic testing')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    if config is None:
        config = {}
    
    if args.parallel:
        config.setdefault("parallel_execution", {})["enabled"] = True
    if args.baselines:
        config.setdefault("performance_baselines", {})["enabled"] = True
    if args.regression:
        config.setdefault("regression_testing", {})["enabled"] = True
    if args.synthetic:
        config.setdefault("synthetic_testing", {})["enabled"] = True
    if args.quick:
        config["query_count"] = 10
        config.setdefault("synthetic_testing", {})["enabled"] = False
    
    # Run tests
    orchestrator = PerformanceTestOrchestrator(config)
    
    try:
        results = await orchestrator.run_enhanced_performance_tests()
        
        # Print final summary
        print_phase("📊 Final Summary")
        summary = results.get("summary", {})
        
        print_success(f"✅ Total execution time: {summary.get('total_execution_time', 0):.2f}s")
        print_info(f"📊 Success rate: {summary.get('success_rate', 0):.1%}")
        print_info(f"⚡ Avg response time: {summary.get('avg_response_time', 0):.2f}s")
        
        if summary.get("regression_tests_total", 0) > 0:
            print_info(f"🔄 Regression tests: {summary.get('regression_tests_passed', 0)}/{summary.get('regression_tests_total', 0)} passed")
        
        if results.get("status") == "completed":
            print_success("🎉 Enhanced performance tests completed successfully!")
            return 0
        else:
            print_error("❌ Enhanced performance tests failed!")
            return 1
            
    except KeyboardInterrupt:
        print_warning("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        print_error(f"\n❌ Tests failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 