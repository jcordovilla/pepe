#!/usr/bin/env python3
"""
Discord Bot Test Runner

Unified test runner for the Discord bot system that provides:
- Core functionality tests
- Integration tests
- Performance benchmarks
- Production readiness assessment
- Comprehensive test reporting

Usage:
    python tests/run_all_tests.py [--core] [--integration] [--performance] [--quick]
    
Examples:
    python tests/run_all_tests.py --quick          # Quick smoke tests
    python tests/run_all_tests.py --core           # Core functionality only
    python tests/run_all_tests.py --integration    # Integration tests only
    python tests/run_all_tests.py                  # All tests (default)
"""

import argparse
import asyncio
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))


class TestResult:
    """Test result container"""
    
    def __init__(self, test_name: str, success: bool, duration: float, 
                 details: Optional[str] = None, error: Optional[str] = None):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.details = details
        self.error = error
        self.timestamp = datetime.now()


class DiscordBotTestRunner:
    """Comprehensive test runner for Discord bot system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.test_results: List[TestResult] = []
        self.start_time = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        from dotenv import load_dotenv
        load_dotenv()
        
        return {
            'has_openai_key': bool(os.getenv('OPENAI_API_KEY')),
            'has_discord_token': bool(os.getenv('DISCORD_TOKEN')),
            'has_guild_id': bool(os.getenv('GUILD_ID')),
            'python_version': sys.version_info,
            'project_root': self.project_root
        }
    
    async def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        self.start_time = time.time()
        
        print("🧪 Discord Bot Comprehensive Test Suite")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Project: {self.project_root}")
        print()
        
        # Determine which tests to run
        if test_types is None:
            test_types = ['quick', 'core', 'integration']
        
        # Environment check
        await self._run_environment_check()
        
        # Run test categories
        for test_type in test_types:
            if test_type == 'quick':
                await self._run_quick_tests()
            elif test_type == 'core':
                await self._run_core_tests()
            elif test_type == 'integration':
                await self._run_integration_tests()
            elif test_type == 'performance':
                await self._run_performance_tests()
        
        # Generate comprehensive report
        report = await self._generate_final_report()
        
        return report
    
    async def _run_environment_check(self):
        """Check test environment requirements"""
        print("🔍 Environment Check")
        print("-" * 30)
        
        start_time = time.time()
        
        checks = [
            ('Python 3.11+', sys.version_info >= (3, 11)),
            ('OpenAI API Key', self.config['has_openai_key']),
            ('Discord Token', self.config['has_discord_token']),
            ('Guild ID', self.config['has_guild_id']),
            ('Project Structure', self._check_project_structure()),
            ('Dependencies', await self._check_dependencies())
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        duration = time.time() - start_time
        
        self.test_results.append(TestResult(
            test_name="Environment Check",
            success=all_passed,
            duration=duration,
            details=f"Checked {len(checks)} requirements"
        ))
        
        if not all_passed:
            print("\n⚠️ Some environment checks failed. Tests may not run correctly.")
        
        print()
    
    async def _run_quick_tests(self):
        """Run quick smoke tests"""
        print("⚡ Quick Smoke Tests")
        print("-" * 30)
        
        quick_tests = [
            ('Import Core Modules', self._test_import_core_modules),
            ('Vector Store Creation', self._test_vector_store_creation),
            ('Agent API Initialization', self._test_agent_api_init),
            ('Basic Configuration', self._test_basic_configuration)
        ]
        
        for test_name, test_func in quick_tests:
            await self._run_single_test(test_name, test_func)
        
        print()
    
    async def _run_core_tests(self):
        """Run core functionality tests"""
        print("🔧 Core Functionality Tests")
        print("-" * 30)
        
        # Run pytest for core tests
        await self._run_pytest_suite(
            "tests/test_discord_bot_core.py",
            "Core Tests",
            markers="not integration"
        )
        
        print()
    
    async def _run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Integration Tests")
        print("-" * 30)
        
        # Run pytest for integration tests
        await self._run_pytest_suite(
            "tests/test_integration.py",
            "Integration Tests",
            markers="integration"
        )
        
        print()
    
    async def _run_performance_tests(self):
        """Run performance benchmarks"""
        print("📊 Performance Tests")
        print("-" * 30)
        
        performance_tests = [
            ('Query Response Time', self._test_query_performance),
            ('Bulk Data Processing', self._test_bulk_processing_performance),
            ('Memory Usage', self._test_memory_usage),
            ('Concurrent Operations', self._test_concurrent_performance)
        ]
        
        for test_name, test_func in performance_tests:
            await self._run_single_test(test_name, test_func)
        
        print()
    
    async def _run_single_test(self, test_name: str, test_func):
        """Run a single test function"""
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            if result.get('success', False):
                print(f"   ✅ {test_name} ({duration:.2f}s)")
                self.test_results.append(TestResult(
                    test_name=test_name,
                    success=True,
                    duration=duration,
                    details=result.get('details')
                ))
            else:
                print(f"   ❌ {test_name} ({duration:.2f}s)")
                self.test_results.append(TestResult(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=result.get('error')
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ {test_name} ({duration:.2f}s) - {str(e)}")
            self.test_results.append(TestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error=str(e)
            ))
    
    async def _run_pytest_suite(self, test_file: str, suite_name: str, markers: str = None):
        """Run pytest test suite"""
        start_time = time.time()
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
        
        if markers:
            cmd.extend(["-m", markers])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse pytest output for details
            output_lines = result.stdout.split('\n')
            test_summary = self._parse_pytest_output(output_lines)
            
            status = "✅" if success else "❌"
            print(f"   {status} {suite_name} ({duration:.2f}s)")
            
            if test_summary:
                print(f"     📊 {test_summary}")
            
            self.test_results.append(TestResult(
                test_name=suite_name,
                success=success,
                duration=duration,
                details=test_summary,
                error=result.stderr if not success else None
            ))
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"   ⏰ {suite_name} TIMEOUT ({duration:.2f}s)")
            self.test_results.append(TestResult(
                test_name=suite_name,
                success=False,
                duration=duration,
                error="Test suite timed out"
            ))
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ {suite_name} ERROR ({duration:.2f}s)")
            self.test_results.append(TestResult(
                test_name=suite_name,
                success=False,
                duration=duration,
                error=str(e)
            ))
    
    def _parse_pytest_output(self, output_lines: List[str]) -> str:
        """Parse pytest output for summary information"""
        for line in output_lines:
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                return line.strip()
            elif line.startswith('=') and 'passed' in line:
                return line.strip('= ')
        
        return "Test completed"
    
    # Individual test implementations
    async def _test_import_core_modules(self) -> Dict[str, Any]:
        """Test importing core modules"""
        try:
            from agentic.interfaces.agent_api import AgentAPI
            from agentic.vectorstore.persistent_store import PersistentVectorStore
            from agentic.memory.conversation_memory import ConversationMemory
            
            return {'success': True, 'details': 'All core modules imported successfully'}
        except Exception as e:
            return {'success': False, 'error': f'Import failed: {e}'}
    
    async def _test_vector_store_creation(self) -> Dict[str, Any]:
        """Test vector store creation"""
        try:
            from agentic.vectorstore.persistent_store import PersistentVectorStore
            
            config = {
                'persist_directory': './tests/temp_vector_test',
                'collection_name': 'test_collection'
            }
            
            store = PersistentVectorStore(config)
            
            # Cleanup
            import shutil
            test_dir = Path('./tests/temp_vector_test')
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            return {'success': True, 'details': 'Vector store created successfully'}
        except Exception as e:
            return {'success': False, 'error': f'Vector store creation failed: {e}'}
    
    async def _test_agent_api_init(self) -> Dict[str, Any]:
        """Test Agent API initialization"""
        try:
            from agentic.interfaces.agent_api import AgentAPI
            
            config = {
                'vector_store': {
                    'persist_directory': './tests/temp_agent_test',
                    'collection_name': 'test_agent_collection'
                },
                'memory': {'db_path': './tests/temp_agent_memory.db'}
            }
            
            api = AgentAPI(config)
            
            # Cleanup
            import shutil
            for path in ['./tests/temp_agent_test', './tests/temp_agent_memory.db']:
                p = Path(path)
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
            
            return {'success': True, 'details': 'Agent API initialized successfully'}
        except Exception as e:
            return {'success': False, 'error': f'Agent API initialization failed: {e}'}
    
    async def _test_basic_configuration(self) -> Dict[str, Any]:
        """Test basic configuration loading"""
        try:
            required_env_vars = ['OPENAI_API_KEY', 'DISCORD_TOKEN', 'GUILD_ID']
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                return {
                    'success': False, 
                    'error': f'Missing environment variables: {", ".join(missing_vars)}'
                }
            
            return {'success': True, 'details': 'All required environment variables present'}
        except Exception as e:
            return {'success': False, 'error': f'Configuration check failed: {e}'}
    
    async def _test_query_performance(self) -> Dict[str, Any]:
        """Test query response performance"""
        try:
            from agentic.interfaces.agent_api import AgentAPI
            
            config = {
                'vector_store': {
                    'persist_directory': './tests/temp_perf_test',
                    'collection_name': 'perf_test'
                }
            }
            
            api = AgentAPI(config)
            
            # Simple performance test
            start_time = time.time()
            result = await api.query(
                query="test performance",
                user_id="perf_test_user",
                context={'platform': 'test'}
            )
            query_time = time.time() - start_time
            
            # Cleanup
            import shutil
            test_dir = Path('./tests/temp_perf_test')
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            if query_time < 5.0:  # Under 5 seconds is good
                return {
                    'success': True, 
                    'details': f'Query completed in {query_time:.2f}s'
                }
            else:
                return {
                    'success': False,
                    'error': f'Query too slow: {query_time:.2f}s'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Performance test failed: {e}'}
    
    async def _test_bulk_processing_performance(self) -> Dict[str, Any]:
        """Test bulk data processing performance"""
        try:
            from agentic.interfaces.agent_api import AgentAPI
            
            config = {
                'vector_store': {
                    'persist_directory': './tests/temp_bulk_test',
                    'collection_name': 'bulk_test'
                }
            }
            
            api = AgentAPI(config)
            
            # Create test data
            test_data = [
                {
                    'id': f'bulk_{i}',
                    'content': f'Test message {i}',
                    'author': 'test_user',
                    'timestamp': '2024-01-01T12:00:00Z'
                }
                for i in range(50)
            ]
            
            # Test bulk processing
            start_time = time.time()
            result = await api.add_documents(test_data, source="bulk_test")
            bulk_time = time.time() - start_time
            
            # Cleanup
            import shutil
            test_dir = Path('./tests/temp_bulk_test')
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            if bulk_time < 15.0 and result.get('success'):  # Under 15 seconds for 50 docs
                return {
                    'success': True,
                    'details': f'Processed 50 documents in {bulk_time:.2f}s'
                }
            else:
                return {
                    'success': False,
                    'error': f'Bulk processing too slow or failed: {bulk_time:.2f}s'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Bulk processing test failed: {e}'}
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        try:
            import psutil
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory usage is reasonable if under 1GB for test process
            if initial_memory < 1024:
                return {
                    'success': True,
                    'details': f'Memory usage: {initial_memory:.1f}MB'
                }
            else:
                return {
                    'success': False,
                    'error': f'High memory usage: {initial_memory:.1f}MB'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Memory test failed: {e}'}
    
    async def _test_concurrent_performance(self) -> Dict[str, Any]:
        """Test concurrent operation performance"""
        try:
            # Simple concurrent test
            async def simple_task(i):
                await asyncio.sleep(0.1)  # Simulate work
                return f"task_{i}"
            
            start_time = time.time()
            tasks = [simple_task(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            concurrent_time = time.time() - start_time
            
            if concurrent_time < 2.0 and len(results) == 10:  # Should complete in under 2s
                return {
                    'success': True,
                    'details': f'10 concurrent tasks in {concurrent_time:.2f}s'
                }
            else:
                return {
                    'success': False,
                    'error': f'Concurrent test too slow: {concurrent_time:.2f}s'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Concurrent test failed: {e}'}
    
    def _check_project_structure(self) -> bool:
        """Check if project structure is correct"""
        required_paths = [
            'agentic',
            'agentic/interfaces',
            'agentic/vectorstore',
            'agentic/memory',
            'main.py',
            'pyproject.toml'
        ]
        
        return all((self.project_root / path).exists() for path in required_paths)
    
    async def _check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        required_modules = [
            'discord',
            'openai', 
            'chromadb',
            'langchain',
            'pytest'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                return False
        
        return True
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Performance statistics
        avg_duration = sum(result.duration for result in self.test_results) / total_tests if total_tests > 0 else 0
        slowest_test = max(self.test_results, key=lambda x: x.duration) if self.test_results else None
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': success_rate
            },
            'performance': {
                'average_test_duration': avg_duration,
                'slowest_test': {
                    'name': slowest_test.test_name if slowest_test else None,
                    'duration': slowest_test.duration if slowest_test else None
                }
            },
            'environment': self.config,
            'test_results': [
                {
                    'name': result.test_name,
                    'success': result.success,
                    'duration': result.duration,
                    'details': result.details,
                    'error': result.error
                }
                for result in self.test_results
            ]
        }
        
        # Display summary
        print("📊 Test Summary")
        print("=" * 60)
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"📋 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print()
        
        # Production readiness assessment
        if success_rate >= 95:
            print("🎉 PRODUCTION READY!")
            print("   All critical systems are functioning correctly.")
        elif success_rate >= 80:
            print("⚠️ MOSTLY READY")
            print("   Most systems working, minor issues detected.")
        else:
            print("❌ NOT READY")
            print("   Critical issues found. Fix before deployment.")
        
        # Failed tests details
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"   • {result.test_name}: {result.error}")
        
        # Save report
        await self._save_test_report(report)
        
        return report
    
    async def _save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        reports_dir = Path("tests/reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Test report saved: {report_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Discord Bot Test Runner")
    parser.add_argument('--quick', action='store_true', help='Run quick smoke tests only')
    parser.add_argument('--core', action='store_true', help='Run core functionality tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    
    args = parser.parse_args()
    
    # Determine test types to run
    test_types = []
    if args.quick:
        test_types.append('quick')
    if args.core:
        test_types.append('core')
    if args.integration:
        test_types.append('integration')
    if args.performance:
        test_types.append('performance')
    
    # If no specific tests requested, run all
    if not test_types:
        test_types = ['quick', 'core', 'integration']
    
    # Run tests
    runner = DiscordBotTestRunner()
    
    try:
        report = asyncio.run(runner.run_all_tests(test_types))
        
        # Exit with appropriate code
        success_rate = report['summary']['success_rate']
        sys.exit(0 if success_rate >= 80 else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 