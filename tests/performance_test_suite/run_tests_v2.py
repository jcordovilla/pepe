#!/usr/bin/env python3
"""
Enhanced Performance Test Runner

Simple runner script for the enhanced performance test suite with all improvements.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.performance_test_suite.orchestrator import PerformanceTestOrchestrator

async def run_enhanced_tests():
    """Run the enhanced performance test suite."""
    print("🚀 Enhanced Performance Test Suite")
    print("=" * 60)
    
    # Configuration for enhanced testing
    config = {
        "database_path": "data/discord_messages.db",
        "query_count": 20,  # Reduced for faster testing
        "save_intermediate_results": True,
        
        # Enable all enhanced features
        "parallel_execution": {
            "enabled": True,
            "max_concurrent": 3,  # Conservative for testing
            "rate_limit_per_second": 5
        },
        "performance_baselines": {
            "enabled": True,
            "baseline_threshold": 0.15,
            "auto_update_baselines": False,
            "baseline_name": "test_baseline"
        },
        "regression_testing": {
            "enabled": True,
            "critical_paths": [
                "server_analysis",
                "semantic_search"
            ]
        },
        "synthetic_testing": {
            "enabled": True,
            "edge_cases": True,
            "stress_scenarios": True,
            "load_testing": False  # Disabled for initial testing
        },
        "resource_monitoring": {
            "enabled": True,
            "monitoring_interval": 1.0
        }
    }
    
    # Create orchestrator
    orchestrator = PerformanceTestOrchestrator(config)
    
    try:
        # Run enhanced tests
        results = await orchestrator.run_enhanced_performance_tests()
        
        # Print results
        if results.get("status") == "completed":
            print("\n🎉 Enhanced tests completed successfully!")
            
            summary = results.get("summary", {})
            print(f"\n📊 Summary:")
            print(f"   Total execution time: {summary.get('total_execution_time', 0):.2f}s")
            print(f"   Total queries: {summary.get('total_queries', 0)}")
            print(f"   Success rate: {summary.get('success_rate', 0):.1%}")
            print(f"   Avg response time: {summary.get('avg_response_time', 0):.2f}s")
            
            if summary.get("regression_tests_total", 0) > 0:
                print(f"   Regression tests: {summary.get('regression_tests_passed', 0)}/{summary.get('regression_tests_total', 0)} passed")
            
            print(f"\n⏱️ Phase times:")
            for phase, time_taken in summary.get("phase_times", {}).items():
                print(f"   {phase}: {time_taken:.2f}s")
            
            return 0
        else:
            print(f"\n❌ Tests failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        return 1

def main():
    """Main entry point."""
    exit_code = asyncio.run(run_enhanced_tests())
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 