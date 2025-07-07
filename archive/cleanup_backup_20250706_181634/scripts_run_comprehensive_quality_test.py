#!/usr/bin/env python3
"""
Run Comprehensive Agent Quality Evaluation

This script runs the comprehensive agent evaluation system that measures:
- Quantitative metrics (scores, response times, success rates)
- Qualitative analysis (strengths, weaknesses, user experience)
- Performance insights (bottlenecks, optimization opportunities)
- Actionable recommendations for improvement

Usage:
    python scripts/run_comprehensive_quality_test.py
    python scripts/run_comprehensive_quality_test.py --detailed
    python scripts/run_comprehensive_quality_test.py --save-all
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from tests.test_comprehensive_agent_quality import ComprehensiveAgentQualityTest


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
        "bold": "\033[1m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def print_comprehensive_report(report):
    """Print comprehensive evaluation report"""
    print_colored("\n" + "="*90, "cyan")
    print_colored("🔍 COMPREHENSIVE AGENT QUALITY EVALUATION RESULTS", "bold")
    print_colored("="*90, "cyan")
    
    # Overall Health
    health_score = report['overall_health_score']
    health_color = "green" if health_score >= 8 else "yellow" if health_score >= 6 else "red"
    print_colored(f"🎯 Overall System Health: {health_score:.1f}/10", health_color)
    
    # Response Time Analysis
    rt_analysis = report['response_time_analysis']
    avg_time = rt_analysis['average']
    time_color = "green" if avg_time <= 3 else "yellow" if avg_time <= 6 else "red"
    print_colored(f"⚡ Response Time - Avg: {avg_time:.1f}s | P95: {rt_analysis['p95']:.1f}s | Max: {rt_analysis['max']:.1f}s", time_color)
    
    # Quality Distribution
    print_colored("\n📊 Quality Distribution:", "yellow")
    quality_dist = report['quality_distribution']
    total_tests = sum(quality_dist.values())
    for quality_level, count in quality_dist.items():
        percentage = (count / total_tests * 100) if total_tests > 0 else 0
        color = "green" if quality_level in ["ResponseQuality.EXCELLENT", "ResponseQuality.GOOD"] else "yellow" if quality_level == "ResponseQuality.ACCEPTABLE" else "red"
        print_colored(f"  {quality_level.split('.')[-1].title()}: {count} ({percentage:.1f}%)", color)
    
    # Performance by Category
    print_colored("\n📋 Performance by Category:", "yellow")
    for category, score in sorted(report['performance_by_category'].items(), key=lambda x: x[1], reverse=True):
        color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
        status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
        category_name = category.replace('_', ' ').title()
        print_colored(f"  {status} {category_name}: {score:.1f}/10", color)
    
    # Performance by Complexity
    if report['performance_by_complexity']:
        print_colored("\n🧠 Performance by Complexity:", "yellow")
        for complexity, score in report['performance_by_complexity'].items():
            color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            print_colored(f"  {complexity.title()}: {score:.1f}/10", color)
    
    # Critical Issues
    if report['critical_issues']:
        print_colored(f"\n❌ Critical Issues ({len(report['critical_issues'])}):", "red")
        for issue in report['critical_issues']:
            print_colored(f"  • {issue}", "red")
    else:
        print_colored("\n✅ No Critical Issues Detected", "green")
    
    # System Bottlenecks
    if report['system_bottlenecks']:
        print_colored(f"\n🚧 System Bottlenecks ({len(report['system_bottlenecks'])}):", "yellow")
        for bottleneck in report['system_bottlenecks']:
            print_colored(f"  • {bottleneck}", "yellow")
    
    # Top Improvement Priorities
    if report['improvement_priorities']:
        print_colored(f"\n🎯 Top Improvement Priorities:", "purple")
        for i, priority in enumerate(report['improvement_priorities'][:3], 1):
            priority_color = "red" if priority['priority_level'] == "high" else "yellow"
            print_colored(f"  {i}. {priority['category'].replace('_', ' ').title()}", priority_color)
            print_colored(f"     Current Score: {priority['current_performance']:.1f}/10", priority_color)
            print_colored(f"     Priority: {priority['priority_level'].upper()}", priority_color)
            print_colored(f"     Impact: {priority['estimated_impact'].title()}", priority_color)
            if priority['recommended_actions']:
                print_colored(f"     Action: {priority['recommended_actions'][0]}", priority_color)
    
    # Technical Recommendations
    if report['technical_recommendations']:
        print_colored(f"\n🔧 Key Technical Recommendations:", "blue")
        for rec in report['technical_recommendations'][:5]:
            print_colored(f"  • {rec}", "blue")
    
    # UX Recommendations
    if report['user_experience_recommendations']:
        print_colored(f"\n👥 User Experience Recommendations:", "blue")
        for rec in report['user_experience_recommendations'][:5]:
            print_colored(f"  • {rec}", "blue")
    
    print_colored("="*90, "cyan")


def print_detailed_insights(report):
    """Print detailed performance insights"""
    print_colored("\n" + "="*90, "purple")
    print_colored("📈 DETAILED PERFORMANCE INSIGHTS", "bold")
    print_colored("="*90, "purple")
    
    # Improvement Priorities with Details
    if report['improvement_priorities']:
        for i, priority in enumerate(report['improvement_priorities'], 1):
            print_colored(f"\n🎯 Priority #{i}: {priority['category'].replace('_', ' ').title()}", "purple")
            print_colored(f"   Current Performance: {priority['current_performance']:.1f}/10", "white")
            print_colored(f"   Priority Level: {priority['priority_level'].upper()}", "white")
            print_colored(f"   Estimated Impact: {priority['estimated_impact'].title()}", "white")
            print_colored(f"   Implementation Complexity: {priority['implementation_complexity'].title()}", "white")
            
            if priority['bottlenecks']:
                print_colored("   🚧 Bottlenecks:", "yellow")
                for bottleneck in priority['bottlenecks']:
                    print_colored(f"     • {bottleneck}", "yellow")
            
            if priority['optimization_opportunities']:
                print_colored("   ⚡ Optimization Opportunities:", "blue")
                for opp in priority['optimization_opportunities']:
                    print_colored(f"     • {opp}", "blue")
            
            if priority['recommended_actions']:
                print_colored("   📋 Recommended Actions:", "green")
                for action in priority['recommended_actions']:
                    print_colored(f"     • {action}", "green")
    
    print_colored("="*90, "purple")


async def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive Agent Quality Evaluation")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed performance insights")
    parser.add_argument("--save-all", action="store_true",
                       help="Save all evaluation data to files")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--min-health-score", type=float, default=6.5,
                       help="Minimum system health score threshold")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print_colored("❌ Error: OPENAI_API_KEY environment variable not set", "red")
        print_colored("This comprehensive evaluation requires OpenAI API access for quality assessment", "yellow")
        return 1
    
    if not args.quiet:
        print_colored("🚀 Starting Comprehensive Agent Quality Evaluation...", "cyan")
        print_colored("This comprehensive test will take several minutes to complete.", "yellow")
        print_colored("Testing quantitative metrics, qualitative analysis, and performance insights...", "blue")
    
    try:
        # Initialize and run comprehensive test
        test_runner = ComprehensiveAgentQualityTest()
        report = await test_runner.run_comprehensive_evaluation()
        
        # Convert dataclass to dict for display
        report_dict = {
            'overall_health_score': report.overall_health_score,
            'performance_by_category': report.performance_by_category,
            'performance_by_complexity': report.performance_by_complexity,
            'response_time_analysis': report.response_time_analysis,
            'quality_distribution': {k.value if hasattr(k, 'value') else str(k): v for k, v in report.quality_distribution.items()},
            'system_bottlenecks': report.system_bottlenecks,
            'critical_issues': report.critical_issues,
            'improvement_priorities': [
                {
                    'category': p.category,
                    'current_performance': p.current_performance,
                    'priority_level': p.priority_level,
                    'estimated_impact': p.estimated_impact,
                    'implementation_complexity': p.implementation_complexity,
                    'bottlenecks': p.bottlenecks,
                    'optimization_opportunities': p.optimization_opportunities,
                    'recommended_actions': p.recommended_actions
                } for p in report.improvement_priorities
            ],
            'technical_recommendations': report.technical_recommendations,
            'user_experience_recommendations': report.user_experience_recommendations
        }
        
        # Print results
        if not args.quiet:
            print_comprehensive_report(report_dict)
            
            if args.detailed:
                print_detailed_insights(report_dict)
        
        # Check health threshold
        if report.overall_health_score < args.min_health_score:
            print_colored(f"\n❌ System health score {report.overall_health_score:.1f} below threshold {args.min_health_score}", "red")
            if report.critical_issues:
                print_colored("Critical issues requiring immediate attention:", "red")
                for issue in report.critical_issues:
                    print_colored(f"  • {issue}", "red")
            return 1
        else:
            if not args.quiet:
                print_colored(f"\n✅ System health score {report.overall_health_score:.1f} meets threshold {args.min_health_score}", "green")
        
        # Summary message
        if not args.quiet:
            acceptable_count = sum(
                count for quality_str, count in report_dict['quality_distribution'].items()
                if 'EXCELLENT' in quality_str or 'GOOD' in quality_str or 'ACCEPTABLE' in quality_str
            )
            total_count = sum(report_dict['quality_distribution'].values())
            acceptable_rate = (acceptable_count / total_count * 100) if total_count > 0 else 0
            
            print_colored(f"📊 Summary: {acceptable_rate:.1f}% acceptable responses, {len(report.critical_issues)} critical issues", "cyan")
            if report.improvement_priorities:
                print_colored(f"🎯 Focus Areas: {', '.join([p.category.replace('_', ' ') for p in report.improvement_priorities[:3]])}", "purple")
        
        return 0
        
    except Exception as e:
        print_colored(f"❌ Comprehensive evaluation failed: {e}", "red")
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
