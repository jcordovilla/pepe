#!/usr/bin/env python3
"""
Performance Test Suite Runner

Simple script to run the comprehensive performance test suite.
Usage: python run_tests.py [--config config.json]
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to the path to import the test suite
sys.path.append(str(Path(__file__).parent))

from tests.performance_test_suite.main_orchestrator import PerformanceTestOrchestrator


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


async def main():
    print("\033[95m\033[1m[DEBUG] Performance test runner started.\033[0m")
    print("🚀 Starting Performance Test Suite...")
    print("=" * 60)
    
    # Configuration for the test suite
    config = {
        "database_path": "data/discord_messages.db",  # Same as production Discord bot
        "bot_api_endpoint": "http://localhost:8000",  # Not used - we use in-process AgentAPI
        "output_directory": "tests/performance_test_suite/data",
        "sample_percentage": 0.15,  # 15% sample of messages for content analysis
        "query_count": 30,  # Updated to 30 queries
        "enable_error_scenarios": True,
        "save_intermediate_results": True,
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
        "evaluation": {
            "use_llama": True,
            "llama_model": "llama3.1:8b"
        },
        "performance": {
            "timeout_per_query": 30,
            "max_concurrent_queries": 1
        },
        "content_analysis": {
            "min_sample_size": 1000,
            "max_sample_size": 5000
        },
        # Ensure we use the same configuration as production
        "use_production_config": True,
        "real_data_mode": True
    }
    
    print("Configuration:")
    print(f"  database_path: {config['database_path']}")
    print(f"  bot_api_endpoint: {config['bot_api_endpoint']}")
    print(f"  output_directory: {config['output_directory']}")
    print(f"  sample_percentage: {config['sample_percentage']}")
    print(f"  query_count: {config['query_count']}")
    print(f"  enable_error_scenarios: {config['enable_error_scenarios']}")
    print(f"  save_intermediate_results: {config['save_intermediate_results']}")
    print()
    
    # Initialize and run the performance test orchestrator
    orchestrator = PerformanceTestOrchestrator(config)
    
    print("Starting comprehensive performance test suite...")
    print(f"Target: {config['query_count']} test queries")
    print(f"Database: {config['database_path']}")
    print(f"Output: {config['output_directory']}")
    print("=" * 60)
    
    try:
        # Run the test suite
        results = await orchestrator.run_performance_tests()
        print(f"\n✅ Test suite completed successfully!")
        print(f"Results summary: {orchestrator.get_test_summary()}")
    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 