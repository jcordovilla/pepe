#!/usr/bin/env python3
"""
Comprehensive test runner for Discord Agent System

Provides structured test execution with different test suites and reporting.
"""
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

class TestRunner:
    """Organized test execution and reporting"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "tests"
        self.results = {}
    
    def run_test_suite(self, name: str, markers: List[str] = None, files: List[str] = None) -> Dict[str, Any]:
        """Run a specific test suite with given markers or files"""
        print(f"\n{'='*60}")
        print(f"RUNNING TEST SUITE: {name.upper()}")
        print(f"{'='*60}")
        
        cmd = ["python", "-m", "pytest"]
        
        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add specific files if specified
        if files:
            cmd.extend(files)
        
        # Add reporting options
        cmd.extend([
            "--tb=short",
            "-v",
            "--color=yes"
        ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
            elapsed = time.time() - start_time
            
            # Parse pytest output for summary
            output_lines = result.stdout.split('\n')
            summary_line = None
            for line in output_lines:
                if 'passed' in line and ('failed' in line or 'error' in line or line.strip().endswith('passed')):
                    summary_line = line.strip()
                    break
            
            suite_result = {
                'name': name,
                'success': result.returncode == 0,
                'duration': elapsed,
                'summary': summary_line or "No summary found",
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Print immediate results
            status = "✅ PASSED" if suite_result['success'] else "❌ FAILED"
            print(f"\n{status} - {name} ({elapsed:.1f}s)")
            if summary_line:
                print(f"Summary: {summary_line}")
            
            return suite_result
            
        except Exception as e:
            return {
                'name': name,
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Exception: {e}"
            }
    
    def run_all_suites(self):
        """Run all test suites in logical order"""
        
        # 1. Fast Unit Tests
        self.results['unit'] = self.run_test_suite(
            "Unit Tests",
            markers=["unit", "not slow", "not performance"]
        )
        
        # 2. Enhanced K Determination Tests  
        self.results['enhanced_k'] = self.run_test_suite(
            "Enhanced K Determination",
            files=["tests/test_enhanced_k_determination.py"]
        )
        
        # 3. Database Integration Tests
        self.results['database'] = self.run_test_suite(
            "Database Integration", 
            files=["tests/test_database_integration.py"]
        )
        
        # 4. Agent Integration Tests
        self.results['agent_integration'] = self.run_test_suite(
            "Agent Integration",
            files=["tests/test_agent_integration.py"]
        )
        
        # 5. Time Parser Tests
        self.results['time_parser'] = self.run_test_suite(
            "Time Parser",
            files=["tests/test_time_parser_comprehensive.py"]
        )
        
        # 6. Summarizer Tests
        self.results['summarizer'] = self.run_test_suite(
            "Summarizer",
            files=["tests/test_summarizer.py"]
        )
        
        # 7. Performance Tests (slower)
        self.results['performance'] = self.run_test_suite(
            "Performance Tests",
            files=["tests/test_performance.py"]
        )
    
    def run_quick_suite(self):
        """Run only fast, essential tests"""
        print("Running Quick Test Suite (Unit + Core Integration)")
        
        self.results['quick_unit'] = self.run_test_suite(
            "Quick Unit Tests",
            markers=["unit", "not slow"]
        )
        
        self.results['quick_enhanced_k'] = self.run_test_suite(
            "Enhanced K Core Tests",
            files=["tests/test_enhanced_k_determination.py"],
        )
    
    def run_integration_suite(self):
        """Run only integration tests"""
        print("Running Integration Test Suite")
        
        self.results['integration'] = self.run_test_suite(
            "All Integration Tests",
            markers=["integration"]
        )
    
    def print_final_report(self):
        """Print comprehensive final report"""
        print(f"\n{'='*80}")
        print("FINAL TEST REPORT")
        print(f"{'='*80}")
        
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r.get('success', False))
        total_time = sum(r.get('duration', 0) for r in self.results.values())
        
        print(f"Total Test Suites: {total_suites}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {total_suites - passed_suites}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Success Rate: {(passed_suites/total_suites)*100:.1f}%")
        
        print(f"\nDetailed Results:")
        for suite_name, result in self.results.items():
            status = "✅" if result.get('success', False) else "❌"
            duration = result.get('duration', 0)
            summary = result.get('summary', 'No summary')
            
            print(f"{status} {suite_name:20} {duration:6.1f}s - {summary}")
        
        # Show failures in detail
        failures = [r for r in self.results.values() if not r.get('success', False)]
        if failures:
            print(f"\n{'='*60}")
            print("FAILURE DETAILS")
            print(f"{'='*60}")
            
            for failure in failures:
                print(f"\n❌ {failure['name']}:")
                if 'error' in failure:
                    print(f"Error: {failure['error']}")
                if failure.get('stderr'):
                    print(f"Stderr: {failure['stderr'][:500]}...")
        
        print(f"\n{'='*80}")
        overall_status = "✅ ALL TESTS PASSED" if passed_suites == total_suites else "❌ SOME TESTS FAILED"
        print(f"{overall_status}")
        print(f"{'='*80}")

def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord Agent System Test Runner")
    parser.add_argument("--suite", choices=["all", "quick", "integration", "performance"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--markers", nargs="+", help="Specific pytest markers to run")
    parser.add_argument("--files", nargs="+", help="Specific test files to run")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.markers or args.files:
            # Custom run
            result = runner.run_test_suite(
                "Custom Tests",
                markers=args.markers,
                files=args.files
            )
            runner.results['custom'] = result
            
        elif args.suite == "quick":
            runner.run_quick_suite()
            
        elif args.suite == "integration":
            runner.run_integration_suite()
            
        elif args.suite == "all":
            runner.run_all_suites()
        
        runner.print_final_report()
        
        # Exit with appropriate code
        failed_suites = [r for r in runner.results.values() if not r.get('success', False)]
        sys.exit(len(failed_suites))
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
