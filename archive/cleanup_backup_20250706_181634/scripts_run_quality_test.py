#!/usr/bin/env python3
"""
Run Agent Response Quality Test

This script runs the comprehensive agent response quality evaluation test
and generates a detailed report.

Usage:
    python scripts/run_quality_test.py
    python scripts/run_quality_test.py --save-report
    python scripts/run_quality_test.py --min-pass-rate 75
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from tests.test_agent_response_quality import AgentResponseQualityTest


def print_colored(text: str, color: str = "white"):
    """Print colored text to terminal"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def print_report_summary(report):
    """Print a nicely formatted report summary"""
    summary = report['summary']
    
    print_colored("\n" + "="*70, "cyan")
    print_colored("🤖 AGENT RESPONSE QUALITY TEST RESULTS", "cyan")
    print_colored("="*70, "cyan")
    
    # Overall results
    pass_rate = summary['pass_rate']
    color = "green" if pass_rate >= 70 else "yellow" if pass_rate >= 50 else "red"
    print_colored(f"📊 Pass Rate: {pass_rate:.1f}%", color)
    
    avg_score = summary['average_score']
    color = "green" if avg_score >= 7 else "yellow" if avg_score >= 6 else "red"
    print_colored(f"⭐ Average Score: {avg_score:.1f}/10", color)
    
    response_time = summary['average_response_time']
    color = "green" if response_time <= 3 else "yellow" if response_time <= 7 else "red"
    print_colored(f"⚡ Average Response Time: {response_time:.1f}s", color)
    
    print_colored(f"✅ Tests Passed: {summary['passed_tests']}", "green")
    print_colored(f"❌ Tests Failed: {summary['failed_tests']}", "red")
    print_colored(f"⏱️  Total Time: {summary['total_time']:.1f}s", "blue")
    
    # Category breakdown
    print_colored("\n📋 Category Performance:", "yellow")
    for category, score in summary['category_averages'].items():
        color = "green" if score >= 7 else "yellow" if score >= 6 else "red"
        category_name = category.replace('_', ' ').title()
        print_colored(f"  {category_name}: {score:.1f}/10", color)
    
    # Failed tests details
    failed_tests = [r for r in report['detailed_results'] if not r['passed']]
    if failed_tests:
        print_colored(f"\n❌ Failed Tests ({len(failed_tests)}):", "red")
        for test in failed_tests[:5]:  # Show first 5 failed tests
            print_colored(f"  • {test['query'][:60]}...", "red")
            print_colored(f"    Category: {test['category']}, Score: {test['evaluation']['overall_score']:.1f}/10", "red")
            if test['error']:
                print_colored(f"    Error: {test['error'][:80]}...", "red")
        
        if len(failed_tests) > 5:
            print_colored(f"  ... and {len(failed_tests) - 5} more", "red")
    
    print_colored("="*70, "cyan")


async def main():
    parser = argparse.ArgumentParser(description="Run Agent Response Quality Test")
    parser.add_argument("--save-report", action="store_true", 
                       help="Save detailed report to file")
    parser.add_argument("--min-pass-rate", type=float, default=70.0,
                       help="Minimum pass rate threshold (default: 70.0)")
    parser.add_argument("--min-avg-score", type=float, default=6.5,
                       help="Minimum average score threshold (default: 6.5)")
    parser.add_argument("--max-response-time", type=float, default=10.0,
                       help="Maximum average response time (default: 10.0s)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print_colored("❌ Error: OPENAI_API_KEY environment variable not set", "red")
        print_colored("Please set your OpenAI API key to run quality evaluations", "yellow")
        return 1
    
    if not args.quiet:
        print_colored("🚀 Starting Agent Response Quality Test...", "cyan")
        print_colored("This may take several minutes to complete.", "yellow")
    
    try:
        # Initialize and run test
        test_runner = AgentResponseQualityTest()
        report = await test_runner.run_comprehensive_test()
        
        # Print results
        if not args.quiet:
            print_report_summary(report)
        
        # Save report if requested
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
            filepath = os.path.join("tests", "reports", filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            print_colored(f"📁 Detailed report saved to: {filepath}", "green")
        
        # Check thresholds
        summary = report['summary']
        exit_code = 0
        
        if summary['pass_rate'] < args.min_pass_rate:
            print_colored(f"❌ Pass rate {summary['pass_rate']:.1f}% below threshold {args.min_pass_rate}%", "red")
            exit_code = 1
        
        if summary['average_score'] < args.min_avg_score:
            print_colored(f"❌ Average score {summary['average_score']:.1f} below threshold {args.min_avg_score}", "red")
            exit_code = 1
        
        if summary['average_response_time'] > args.max_response_time:
            print_colored(f"❌ Response time {summary['average_response_time']:.1f}s above threshold {args.max_response_time}s", "red")
            exit_code = 1
        
        if exit_code == 0 and not args.quiet:
            print_colored("🎉 All quality thresholds met!", "green")
        
        return exit_code
        
    except Exception as e:
        print_colored(f"❌ Test failed with error: {e}", "red")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
