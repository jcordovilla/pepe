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


async def run_tests(config: dict = None):
    """Run the performance test suite."""
    print("🚀 Starting Performance Test Suite...")
    print("=" * 60)
    
    # Default configuration
    default_config = {
        "database_path": "data/discord_messages.db",
        "bot_api_endpoint": "http://localhost:8000",
        "output_directory": "tests/performance_test_suite/data",
        "sample_percentage": 0.15,
        "query_count": 50,
        "enable_error_scenarios": True,
        "save_intermediate_results": True
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    print("Configuration:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create orchestrator
    orchestrator = PerformanceTestOrchestrator(default_config)
    
    try:
        # Run the complete test suite
        results = await orchestrator.run_complete_test_suite()
        
        # Get and display summary
        summary = orchestrator.get_test_summary()
        
        print("\n" + "=" * 60)
        print("✅ PERFORMANCE TEST SUITE - EXECUTION COMPLETE")
        print("=" * 60)
        print(f"Status: {summary['status']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Queries: {summary['summary']['total_queries']}")
        print(f"Successful Responses: {summary['summary']['successful_responses']}")
        print(f"Average Quality Score: {summary['summary']['average_quality_score']:.2f}")
        print(f"System Reliability: {summary['summary']['system_reliability']:.2f}")
        
        print("\n🔍 Key Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\n💡 Top Recommendations:")
        for rec in summary['top_recommendations']:
            print(f"  • {rec}")
        
        print("\n" + "=" * 60)
        print("📊 Test suite execution completed successfully!")
        print("📁 Check the generated reports for detailed analysis and recommendations.")
        print("📂 Reports saved in: tests/performance_test_suite/data/")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Test suite execution failed: {e}")
        return False
    
    finally:
        # Cleanup
        await orchestrator.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Performance Test Suite")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--database", 
        type=str, 
        help="Path to Discord messages database"
    )
    parser.add_argument(
        "--bot-endpoint", 
        type=str, 
        help="Bot API endpoint URL"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    if args.database:
        config["database_path"] = args.database
    if args.bot_endpoint:
        config["bot_api_endpoint"] = args.bot_endpoint
    if args.output_dir:
        config["output_directory"] = args.output_dir
    
    # Run tests
    success = asyncio.run(run_tests(config))
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 