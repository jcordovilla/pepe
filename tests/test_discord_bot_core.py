"""
Discord Bot Core Functionality Tests

Comprehensive test suite focused on the Discord bot's essential operations:
- Discord command processing (/pepe)
- Vector store operations
- Agent system functionality
- Resource management
- Forum channel support
- Sync operations

This replaces multiple scattered test files with a focused, production-ready test suite.
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

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from agentic.interfaces.agent_api import AgentAPI
from agentic.vectorstore.persistent_store import PersistentVectorStore
from agentic.memory.conversation_memory import ConversationMemory
from agentic.analytics.query_answer_repository import QueryAnswerRepository


class DiscordBotCoreTests:
    """Core Discord bot functionality tests"""
    
    def __init__(self):
        self.config = self._load_test_config()
        self.test_data_dir = Path("tests/test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        from dotenv import load_dotenv
        load_dotenv()
        
        return {
            'vector_store': {
                'persist_directory': './tests/test_data/chromadb_test',
                'collection_name': 'test_discord_messages',
                'embedding_model': 'msmarco-distilbert-base-v4',
            'embedding_type': 'sentence_transformers'
            },
            'memory': {
                'db_path': './tests/test_data/test_conversation_memory.db'
            },
            'analytics': {
                'db_path': './tests/test_data/test_analytics.db'
            },
            'discord': {
                'token': os.getenv('DISCORD_TOKEN'),
                'guild_id': os.getenv('GUILD_ID')
            }
        }


@pytest.mark.asyncio
class TestVectorStoreOperations:
    """Test vector store core operations"""
    
    @pytest.fixture
    async def vector_store(self):
        """Create test vector store"""
        test = DiscordBotCoreTests()
        store = PersistentVectorStore(test.config['vector_store'])
        yield store
        # Cleanup after test
        import shutil
        test_db_path = Path(test.config['vector_store']['persist_directory'])
        if test_db_path.exists():
            shutil.rmtree(test_db_path)
    
    async def test_vector_store_initialization(self, vector_store):
        """Test vector store initializes correctly"""
        health = await vector_store.health_check()
        assert health['status'] == 'healthy'
        assert 'collection_name' in health
    
    async def test_add_discord_messages(self, vector_store):
        """Test adding Discord messages to vector store"""
        test_messages = [
            {
                'id': '123456789',
                'content': 'This is a test message about Python programming',
                'author': 'test_user',
                'channel': 'general',
                'timestamp': '2024-01-01T12:00:00Z',
                'metadata': {'channel_id': '987654321'}
            },
            {
                'id': '123456790',
                'content': 'Another message discussing machine learning algorithms',
                'author': 'another_user',
                'channel': 'ai-discussion',
                'timestamp': '2024-01-01T12:30:00Z',
                'metadata': {'channel_id': '987654322'}
            }
        ]
        
        # Add messages
        result = await vector_store.add_documents(test_messages)
        assert result['success'] is True
        assert result['added_count'] == 2
        
        # Verify messages were added
        stats = await vector_store.get_stats()
        assert stats['total_documents'] >= 2
    
    async def test_search_discord_messages(self, vector_store):
        """Test searching Discord messages"""
        # First add test data
        test_messages = [
            {
                'id': '111',
                'content': 'Discussion about Python FastAPI framework',
                'author': 'dev1',
                'channel': 'development',
                'timestamp': '2024-01-01T10:00:00Z'
            },
            {
                'id': '222',
                'content': 'React component optimization techniques',
                'author': 'frontend_dev',
                'channel': 'frontend',
                'timestamp': '2024-01-01T11:00:00Z'
            }
        ]
        
        await vector_store.add_documents(test_messages)
        
        # Search for Python-related content
        results = await vector_store.search("Python FastAPI", limit=5)
        assert len(results) > 0
        assert any('Python' in result.get('content', '') for result in results)
    
    async def test_filter_by_channel(self, vector_store):
        """Test filtering messages by channel"""
        # Add test data with different channels
        test_messages = [
            {
                'id': '301',
                'content': 'Channel specific message',
                'author': 'user1',
                'channel': 'specific-channel',
                'timestamp': '2024-01-01T12:00:00Z',
                'metadata': {'channel_id': '12345'}
            },
            {
                'id': '302',
                'content': 'Another channel message',
                'author': 'user2',
                'channel': 'other-channel',
                'timestamp': '2024-01-01T12:00:00Z',
                'metadata': {'channel_id': '67890'}
            }
        ]
        
        await vector_store.add_documents(test_messages)
        
        # Search with channel filter
        results = await vector_store.search(
            "message",
            filters={'channel': 'specific-channel'},
            limit=5
        )
        
        assert len(results) > 0
        assert all(result.get('channel') == 'specific-channel' for result in results)


@pytest.mark.asyncio
class TestAgentAPI:
    """Test Agent API functionality"""
    
    @pytest.fixture
    async def agent_api(self):
        """Create test agent API"""
        test = DiscordBotCoreTests()
        api = AgentAPI(test.config)
        yield api
        # Cleanup after test
        import shutil
        for db_path in [test.config['vector_store']['persist_directory'],
                       test.config['memory']['db_path'],
                       test.config['analytics']['db_path']]:
            path = Path(db_path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    async def test_agent_api_health_check(self, agent_api):
        """Test agent API health check"""
        health = await agent_api.health_check()
        assert health['status'] in ['healthy', 'degraded']
        assert 'components' in health
    
    async def test_basic_query_processing(self, agent_api):
        """Test basic query processing"""
        # Simple query test
        result = await agent_api.query(
            query="What are your capabilities?",
            user_id="test_user",
            context={"platform": "discord", "channel": "test"}
        )
        
        assert result['success'] is True
        assert 'answer' in result
        assert len(result['answer']) > 0
        assert 'processing_time' in result
    
    async def test_search_query(self, agent_api):
        """Test search query functionality"""
        # Add some test data first
        test_messages = [
            {
                'id': '401',
                'content': 'Discussion about Discord bot development using Python',
                'author': 'bot_developer',
                'channel': 'bot-development',
                'timestamp': '2024-01-01T15:00:00Z'
            }
        ]
        
        await agent_api.add_documents(test_messages, source="test")
        
        # Search for the content
        result = await agent_api.query(
            query="Find messages about Discord bot development",
            user_id="test_user",
            context={"platform": "discord", "channel": "test"}
        )
        
        assert result['success'] is True
        assert 'Discord' in result['answer'] or 'bot' in result['answer']
    
    async def test_error_handling(self, agent_api):
        """Test error handling in agent API"""
        # Test with invalid/empty query
        result = await agent_api.query(
            query="",
            user_id="test_user",
            context={}
        )
        
        # Should handle gracefully
        assert 'success' in result
        if not result['success']:
            assert 'error' in result


@pytest.mark.asyncio
class TestDiscordBotCommands:
    """Test Discord bot command processing"""
    
    @pytest.fixture
    async def bot_interface(self):
        """Create test bot interface"""
        test = DiscordBotCoreTests()
        
        # Mock Discord bot interface for testing
        class MockDiscordInterface:
            def __init__(self, config):
                self.config = config
                self.agent_api = AgentAPI(config)
            
            async def process_pepe_command(self, query: str, user_id: str, channel_id: str, context: Dict):
                """Mock /pepe command processing"""
                return await self.agent_api.query(
                    query=query,
                    user_id=user_id,
                    context={**context, 'channel_id': channel_id, 'platform': 'discord'}
                )
        
        interface = MockDiscordInterface(test.config)
        yield interface
        
        # Cleanup
        import shutil
        for db_path in [test.config['vector_store']['persist_directory'],
                       test.config['memory']['db_path'],
                       test.config['analytics']['db_path']]:
            path = Path(db_path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    async def test_pepe_command_basic_query(self, bot_interface):
        """Test basic /pepe command query"""
        result = await bot_interface.process_pepe_command(
            query="What can you help me with?",
            user_id="test_user_123",
            channel_id="test_channel_456",
            context={"guild_id": "test_guild_789"}
        )
        
        assert result['success'] is True
        assert 'answer' in result
        assert len(result['answer']) > 0
    
    async def test_pepe_command_search_query(self, bot_interface):
        """Test /pepe command with search functionality"""
        # Add test data
        test_messages = [
            {
                'id': '501',
                'content': 'Important announcement about project deadline',
                'author': 'project_manager',
                'channel': 'announcements',
                'timestamp': '2024-01-01T09:00:00Z'
            }
        ]
        
        await bot_interface.agent_api.add_documents(test_messages, source="test")
        
        # Search query
        result = await bot_interface.process_pepe_command(
            query="Find announcements about project deadlines",
            user_id="test_user_123",
            channel_id="test_channel_456",
            context={"guild_id": "test_guild_789"}
        )
        
        assert result['success'] is True
        assert 'project' in result['answer'].lower() or 'deadline' in result['answer'].lower()
    
    async def test_pepe_command_performance(self, bot_interface):
        """Test /pepe command performance"""
        start_time = time.time()
        
        result = await bot_interface.process_pepe_command(
            query="Simple test query",
            user_id="test_user_123",
            channel_id="test_channel_456",
            context={}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert result['success'] is True
        assert response_time < 10.0  # Should respond within 10 seconds
        
        if 'processing_time' in result:
            assert result['processing_time'] < 8.0  # Internal processing under 8 seconds


@pytest.mark.asyncio  
class TestForumChannelSupport:
    """Test forum channel functionality"""
    
    @pytest.fixture
    async def forum_test_setup(self):
        """Setup for forum channel tests"""
        test = DiscordBotCoreTests()
        
        # Mock forum channel data
        forum_data = {
            'forum_channels': [
                {
                    'id': '789456123',
                    'name': 'help-forum',
                    'threads': [
                        {
                            'id': '111222333',
                            'name': 'Python debugging help',
                            'messages': [
                                {
                                    'id': '555666777',
                                    'content': 'I need help debugging my Python script',
                                    'author': 'newbie_coder',
                                    'timestamp': '2024-01-01T14:00:00Z'
                                },
                                {
                                    'id': '555666778',
                                    'content': 'Try using print statements to trace execution',
                                    'author': 'helpful_dev',
                                    'timestamp': '2024-01-01T14:15:00Z'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        yield forum_data, test.config
        
        # Cleanup
        import shutil
        test_db_path = Path(test.config['vector_store']['persist_directory'])
        if test_db_path.exists():
            shutil.rmtree(test_db_path)
    
    async def test_forum_thread_processing(self, forum_test_setup):
        """Test processing forum thread messages"""
        forum_data, config = forum_test_setup
        
        vector_store = PersistentVectorStore(config['vector_store'])
        
        # Process forum thread messages
        forum_channel = forum_data['forum_channels'][0]
        thread = forum_channel['threads'][0]
        
        # Convert thread messages to vector store format
        documents = []
        for message in thread['messages']:
            documents.append({
                'id': message['id'],
                'content': message['content'],
                'author': message['author'],
                'timestamp': message['timestamp'],
                'channel': forum_channel['name'],
                'thread_name': thread['name'],
                'metadata': {
                    'channel_id': forum_channel['id'],
                    'thread_id': thread['id'],
                    'is_forum_thread': True
                }
            })
        
        # Add to vector store
        result = await vector_store.add_documents(documents)
        assert result['success'] is True
        assert result['added_count'] == 2
        
        # Search for forum content
        search_results = await vector_store.search("Python debugging", limit=5)
        assert len(search_results) > 0
        
        # Verify forum thread metadata
        forum_result = search_results[0]
        assert forum_result.get('thread_name') == 'Python debugging help'
        assert forum_result.get('metadata', {}).get('is_forum_thread') is True
    
    async def test_forum_thread_search_filtering(self, forum_test_setup):
        """Test searching within specific forum threads"""
        forum_data, config = forum_test_setup
        
        vector_store = PersistentVectorStore(config['vector_store'])
        agent_api = AgentAPI(config)
        
        # Add forum data
        forum_channel = forum_data['forum_channels'][0]
        thread = forum_channel['threads'][0]
        
        documents = []
        for message in thread['messages']:
            documents.append({
                'id': message['id'],
                'content': message['content'],
                'author': message['author'],
                'timestamp': message['timestamp'],
                'channel': forum_channel['name'],
                'thread_name': thread['name'],
                'metadata': {
                    'thread_id': thread['id']
                }
            })
        
        await vector_store.add_documents(documents)
        
        # Search with thread filter
        query_result = await agent_api.query(
            query="Find debugging help in Python thread",
            user_id="test_user",
            context={
                'platform': 'discord',
                'filter_thread_id': thread['id']
            }
        )
        
        assert query_result['success'] is True
        assert 'debugging' in query_result['answer'].lower() or 'python' in query_result['answer'].lower()


@pytest.mark.asyncio
class TestSyncOperations:
    """Test data sync operations"""
    
    @pytest.fixture
    async def sync_test_setup(self):
        """Setup for sync operation tests"""
        test = DiscordBotCoreTests()
        
        # Create test message files
        test_data_dir = Path("tests/test_data/fetched_messages")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample message data
        test_messages = [
            {
                'id': '601',
                'content': 'Sample Discord message for sync testing',
                'author': 'sync_test_user',
                'channel': 'sync-test-channel',
                'timestamp': '2024-01-01T16:00:00Z'
            },
            {
                'id': '602', 
                'content': 'Another message for comprehensive sync testing',
                'author': 'another_sync_user',
                'channel': 'sync-test-channel',
                'timestamp': '2024-01-01T16:30:00Z'
            }
        ]
        
        # Save test messages to file
        test_file = test_data_dir / '123456_789012_messages.json'
        with open(test_file, 'w') as f:
            json.dump(test_messages, f)
        
        yield test_data_dir, test.config
        
        # Cleanup
        import shutil
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir.parent)
    
    async def test_message_file_processing(self, sync_test_setup):
        """Test processing Discord message files"""
        test_data_dir, config = sync_test_setup
        
        agent_api = AgentAPI(config)
        
        # Process the test message file
        test_file = test_data_dir / '123456_789012_messages.json'
        
        with open(test_file, 'r') as f:
            messages = json.load(f)
        
        # Add messages to system
        result = await agent_api.add_documents(messages, source="sync_test")
        
        assert result['success'] is True
        assert result['added_count'] == 2
        
        # Verify messages can be searched
        search_result = await agent_api.query(
            query="Find sync testing messages",
            user_id="test_user",
            context={'platform': 'discord'}
        )
        
        assert search_result['success'] is True
        assert 'sync' in search_result['answer'].lower()
    
    async def test_sync_statistics_calculation(self, sync_test_setup):
        """Test sync statistics calculation"""
        test_data_dir, config = sync_test_setup
        
        # Calculate stats from test data
        json_files = list(test_data_dir.glob('*.json'))
        
        stats = {
            'total_files': len(json_files),
            'total_messages': 0,
            'channels_found': set()
        }
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                messages = json.load(f)
            
            stats['total_messages'] += len(messages)
            
            # Extract channel from filename
            filename_parts = json_file.stem.split('_')
            if len(filename_parts) >= 2:
                channel_id = filename_parts[1]
                stats['channels_found'].add(channel_id)
        
        assert stats['total_files'] == 1
        assert stats['total_messages'] == 2
        assert len(stats['channels_found']) == 1
        assert '789012' in stats['channels_found']


@pytest.mark.asyncio
class TestResourceManagement:
    """Test resource management functionality"""
    
    @pytest.fixture
    async def resource_test_setup(self):
        """Setup for resource management tests"""
        test = DiscordBotCoreTests()
        
        # Create test resource database
        import sqlite3
        db_path = Path("tests/test_data/test_resources.db")
        db_path.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create resources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                domain TEXT,
                title TEXT,
                description TEXT,
                quality_score REAL,
                quality_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test resources
        test_resources = [
            ('https://arxiv.org/abs/2024.01234', 'arxiv.org', 'AI Research Paper', 
             'Advanced machine learning techniques', 9.2, 'excellent'),
            ('https://github.com/user/project', 'github.com', 'Open Source Project',
             'Python Discord bot framework', 8.5, 'high'),
            ('https://docs.python.org/3/tutorial/', 'docs.python.org', 'Python Tutorial',
             'Official Python documentation', 9.0, 'excellent')
        ]
        
        cursor.executemany('''
            INSERT INTO resources (url, domain, title, description, quality_score, quality_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', test_resources)
        
        conn.commit()
        conn.close()
        
        yield db_path, test.config
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
    
    async def test_resource_database_access(self, resource_test_setup):
        """Test resource database access"""
        db_path, config = resource_test_setup
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query resources
        cursor.execute('SELECT COUNT(*) FROM resources')
        count = cursor.fetchone()[0]
        assert count == 3
        
        # Query high-quality resources
        cursor.execute('SELECT * FROM resources WHERE quality_score >= 9.0')
        high_quality = cursor.fetchall()
        assert len(high_quality) == 2
        
        conn.close()
    
    async def test_resource_integration_with_search(self, resource_test_setup):
        """Test resource integration with search functionality"""
        db_path, config = resource_test_setup
        
        # This test would verify that resources are properly integrated
        # with the search and recommendation system
        
        agent_api = AgentAPI(config)
        
        # Simulate a query that should reference resources
        result = await agent_api.query(
            query="Find resources about Python programming",
            user_id="test_user",
            context={'platform': 'discord', 'include_resources': True}
        )
        
        assert result['success'] is True
        # In a full implementation, this would check if relevant resources
        # are included in the response


def run_core_tests():
    """Run all core Discord bot tests"""
    print("🧪 Running Discord Bot Core Tests")
    print("=" * 50)
    
    # Configure pytest arguments
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "-x",  # Stop on first failure
        "--tb=short",  # Short traceback format
        "--asyncio-mode=auto"  # Auto async mode
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All core tests passed!")
        print("🎉 Discord bot system is production-ready!")
    else:
        print("\n❌ Some tests failed!")
        print("🔧 Please fix issues before deployment")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_core_tests()
    sys.exit(exit_code) 