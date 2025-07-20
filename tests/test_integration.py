"""
Discord Bot Integration Tests

End-to-end integration tests focusing on real-world Discord bot scenarios:
- Complete workflow from Discord message to response
- Admin CLI operations integration
- Resource management integration
- Performance and reliability testing

These tests ensure all components work together correctly in production scenarios.
"""

import asyncio
import json
import os
import sys
import time
import pytest
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.interfaces.agent_api import AgentAPI
from agentic.vectorstore.persistent_store import PersistentVectorStore


class IntegrationTestConfig:
    """Configuration for integration tests"""
    
    def __init__(self):
        self.test_data_dir = Path("tests/integration_test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        self.config = {
            'vector_store': {
                'persist_directory': str(self.test_data_dir / 'chromadb_integration'),
                'collection_name': 'integration_test_messages',
                'embedding_model': 'text-embedding-3-small'
            },
            'memory': {
                'db_path': str(self.test_data_dir / 'integration_memory.db')
            },
            'analytics': {
                'db_path': str(self.test_data_dir / 'integration_analytics.db')
            },
            'discord': {
                'token': os.getenv('DISCORD_TOKEN'),
                'guild_id': os.getenv('GUILD_ID')
            }
        }
    
    def cleanup(self):
        """Clean up test data"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Test complete Discord bot workflow"""
    
    @pytest.fixture
    async def integration_setup(self):
        """Setup for integration tests"""
        config = IntegrationTestConfig()
        
        # Create sample Discord data
        sample_messages = [
            {
                'id': '1001',
                'content': 'How do I implement async/await in Python?',
                'author': 'python_learner',
                'channel': 'help',
                'timestamp': '2024-01-01T10:00:00Z',
                'metadata': {'channel_id': '123456', 'guild_id': '789012'}
            },
            {
                'id': '1002',
                'content': 'async def my_function(): await some_operation()',
                'author': 'helpful_dev',
                'channel': 'help',
                'timestamp': '2024-01-01T10:05:00Z',
                'metadata': {'channel_id': '123456', 'guild_id': '789012'}
            },
            {
                'id': '1003',
                'content': 'What are the best practices for Discord bot development?',
                'author': 'bot_developer',
                'channel': 'bot-dev',
                'timestamp': '2024-01-01T11:00:00Z',
                'metadata': {'channel_id': '234567', 'guild_id': '789012'}
            },
            {
                'id': '1004',
                'content': 'Use discord.py, implement proper error handling, and rate limiting',
                'author': 'senior_dev',
                'channel': 'bot-dev',
                'timestamp': '2024-01-01T11:15:00Z',
                'metadata': {'channel_id': '234567', 'guild_id': '789012'}
            }
        ]
        
        # Save test messages
        messages_dir = config.test_data_dir / 'fetched_messages'
        messages_dir.mkdir(exist_ok=True)
        
        with open(messages_dir / '789012_123456_messages.json', 'w') as f:
            json.dump([sample_messages[0], sample_messages[1]], f)
        
        with open(messages_dir / '789012_234567_messages.json', 'w') as f:
            json.dump([sample_messages[2], sample_messages[3]], f)
        
        yield config, sample_messages
        
        # Cleanup
        config.cleanup()
    
    async def test_end_to_end_query_workflow(self, integration_setup):
        """Test complete end-to-end query workflow"""
        config, sample_messages = integration_setup
        
        # Step 1: Initialize system
        agent_api = AgentAPI(config.config)
        
        # Step 2: Add sample data
        result = await agent_api.add_documents(sample_messages, source="integration_test")
        assert result['success'] is True
        assert result['added_count'] == 4
        
        # Step 3: Process user query
        query_result = await agent_api.query(
            query="How do I use async/await in Python?",
            user_id="integration_test_user",
            context={
                'platform': 'discord',
                'channel_id': '123456',
                'guild_id': '789012'
            }
        )
        
        # Step 4: Verify response
        assert query_result['success'] is True
        assert 'answer' in query_result
        assert len(query_result['answer']) > 0
        
        # Should contain relevant information about async/await
        answer_lower = query_result['answer'].lower()
        assert any(keyword in answer_lower for keyword in ['async', 'await', 'python'])
        
        # Step 5: Verify performance
        assert query_result.get('processing_time', 0) < 10.0  # Under 10 seconds
    
    async def test_multi_channel_search_integration(self, integration_setup):
        """Test searching across multiple channels"""
        config, sample_messages = integration_setup
        
        agent_api = AgentAPI(config.config)
        await agent_api.add_documents(sample_messages, source="integration_test")
        
        # Search for bot development across channels
        result = await agent_api.query(
            query="Find discussions about bot development best practices",
            user_id="integration_test_user",
            context={'platform': 'discord'}
        )
        
        assert result['success'] is True
        answer_lower = result['answer'].lower()
        assert any(keyword in answer_lower for keyword in ['bot', 'development', 'discord'])
    
    async def test_conversation_memory_integration(self, integration_setup):
        """Test conversation memory across multiple queries"""
        config, sample_messages = integration_setup
        
        agent_api = AgentAPI(config.config)
        await agent_api.add_documents(sample_messages, source="integration_test")
        
        user_id = "memory_test_user"
        
        # First query
        result1 = await agent_api.query(
            query="What is async/await in Python?",
            user_id=user_id,
            context={'platform': 'discord'}
        )
        assert result1['success'] is True
        
        # Follow-up query (should remember context)
        result2 = await agent_api.query(
            query="Can you give me an example?",
            user_id=user_id,
            context={'platform': 'discord'}
        )
        assert result2['success'] is True
        
        # The follow-up should understand context from previous query
        answer_lower = result2['answer'].lower()
        assert any(keyword in answer_lower for keyword in ['async', 'await', 'example'])


@pytest.mark.integration
@pytest.mark.asyncio
class TestAdminCLIIntegration:
    """Test admin CLI integration with core system"""
    
    @pytest.fixture
    async def cli_test_setup(self):
        """Setup for CLI integration tests"""
        config = IntegrationTestConfig()
        
        # Create test environment
        test_env = os.environ.copy()
        test_env['PYTHONPATH'] = str(Path.cwd())
        
        yield config, test_env
        
        # Cleanup
        config.cleanup()
    
    async def test_admin_setup_command(self, cli_test_setup):
        """Test admin setup command integration"""
        config, test_env = cli_test_setup
        
        # Create temporary admin script for testing
        admin_script = f"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, '{Path.cwd()}')

from pepe_admin_simplified import PepeAdminSimplified

async def test_setup():
    admin = PepeAdminSimplified()
    # Mock args
    class MockArgs:
        pass
    
    try:
        result = await admin.setup(MockArgs())
        print(f"SETUP_RESULT:{result}")
        return result
    except Exception as e:
        print(f"SETUP_ERROR:{e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_setup())
    sys.exit(0 if result else 1)
"""
        
        # Write and execute test script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(admin_script)
            temp_script = f.name
        
        try:
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, env=test_env, cwd=str(Path.cwd()))
            
            # Check if setup was successful
            assert result.returncode == 0 or "SETUP_RESULT:True" in result.stdout
            
        finally:
            os.unlink(temp_script)
    
    async def test_admin_info_command_integration(self, cli_test_setup):
        """Test admin info command integration"""
        config, test_env = cli_test_setup
        
        # Initialize system first
        agent_api = AgentAPI(config.config)
        
        # Add some test data
        test_messages = [
            {
                'id': '2001',
                'content': 'Test message for admin info integration',
                'author': 'test_user',
                'channel': 'test',
                'timestamp': '2024-01-01T12:00:00Z'
            }
        ]
        
        await agent_api.add_documents(test_messages, source="admin_test")
        
        # Test info command (simplified version)
        admin_script = f"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, '{Path.cwd()}')

from pepe_admin_simplified import PepeAdminSimplified

async def test_info():
    admin = PepeAdminSimplified()
    # Mock args
    class MockArgs:
        pass
    
    try:
        result = await admin.info(MockArgs())
        print("INFO_COMMAND_COMPLETED")
        return True
    except Exception as e:
        print(f"INFO_ERROR:{e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_info())
    sys.exit(0 if result else 1)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(admin_script)
            temp_script = f.name
        
        try:
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, env=test_env, timeout=30)
            
            # Should complete without errors
            assert "INFO_COMMAND_COMPLETED" in result.stdout or result.returncode == 0
            
        finally:
            os.unlink(temp_script)


@pytest.mark.integration
@pytest.mark.asyncio
class TestPerformanceIntegration:
    """Test performance under realistic load"""
    
    @pytest.fixture
    async def performance_setup(self):
        """Setup for performance tests"""
        config = IntegrationTestConfig()
        
        # Create larger dataset for performance testing
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                'id': f'perf_{i}',
                'content': f'Performance test message {i} discussing various topics like Python, Discord, and bots',
                'author': f'user_{i % 10}',
                'channel': f'channel_{i % 5}',
                'timestamp': f'2024-01-01T{10 + i // 10:02d}:{i % 60:02d}:00Z',
                'metadata': {'test_data': True}
            })
        
        yield config, large_dataset
        
        # Cleanup
        config.cleanup()
    
    async def test_bulk_data_processing_performance(self, performance_setup):
        """Test performance with bulk data processing"""
        config, large_dataset = performance_setup
        
        agent_api = AgentAPI(config.config)
        
        # Measure bulk insert performance
        start_time = time.time()
        result = await agent_api.add_documents(large_dataset, source="performance_test")
        insert_time = time.time() - start_time
        
        assert result['success'] is True
        assert result['added_count'] == 100
        assert insert_time < 30.0  # Should complete within 30 seconds
        
        print(f"📊 Bulk insert performance: {insert_time:.2f}s for 100 documents")
    
    async def test_concurrent_query_performance(self, performance_setup):
        """Test performance under concurrent queries"""
        config, large_dataset = performance_setup
        
        agent_api = AgentAPI(config.config)
        await agent_api.add_documents(large_dataset, source="performance_test")
        
        # Define test queries
        test_queries = [
            "Find messages about Python",
            "Show me bot development discussions",
            "What are users saying about Discord?",
            "Find recent performance tests",
            "Search for channel discussions"
        ]
        
        # Run concurrent queries
        async def run_query(query, user_id):
            return await agent_api.query(
                query=query,
                user_id=user_id,
                context={'platform': 'discord'}
            )
        
        start_time = time.time()
        
        # Run 10 concurrent queries
        tasks = []
        for i in range(10):
            query = test_queries[i % len(test_queries)]
            task = run_query(query, f"concurrent_user_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        # Verify all queries succeeded
        successful_queries = sum(1 for result in results if isinstance(result, dict) and result.get('success'))
        assert successful_queries >= 8  # At least 80% success rate
        assert concurrent_time < 20.0  # Complete within 20 seconds
        
        print(f"📊 Concurrent query performance: {concurrent_time:.2f}s for 10 queries")
        print(f"📊 Success rate: {successful_queries}/10 ({successful_queries * 10}%)")
    
    async def test_memory_usage_stability(self, performance_setup):
        """Test memory usage remains stable"""
        config, large_dataset = performance_setup
        
        import psutil
        import os
        
        agent_api = AgentAPI(config.config)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        for batch in range(5):
            batch_data = large_dataset[batch * 20:(batch + 1) * 20]
            await agent_api.add_documents(batch_data, source=f"memory_test_batch_{batch}")
            
            # Run some queries
            for i in range(3):
                await agent_api.query(
                    query=f"Test query {batch}_{i}",
                    user_id=f"memory_test_user_{batch}_{i}",
                    context={'platform': 'discord'}
                )
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"📊 Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"


def run_integration_tests():
    """Run all integration tests"""
    print("🔗 Running Discord Bot Integration Tests")
    print("=" * 60)
    
    # Configure pytest for integration tests
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "-m", "integration",  # Only integration tests
        "--tb=short",  # Short traceback format
        "--asyncio-mode=auto",  # Auto async mode
        "-x"  # Stop on first failure
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All integration tests passed!")
        print("🎉 Discord bot integration is working correctly!")
    else:
        print("\n❌ Some integration tests failed!")
        print("🔧 Please fix integration issues before deployment")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code) 