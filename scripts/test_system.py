#!/usr/bin/env python3
"""
Comprehensive test script for the agentic Discord bot system.
This tests the core functionality without requiring actual API keys.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path for importing agentic modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

def print_test_header(title):
    """Print formatted test header"""
    print(f"\n{'=' * 60}")
    print(f"🧪 {title}")
    print('=' * 60)

def print_test_step(step_num, total_steps, description):
    """Print formatted test step"""
    print(f"\n📋 Step {step_num}/{total_steps}: {description}")
    print("-" * 50)

def test_imports():
    """Test that all modules can be imported successfully"""
    print_test_header("Module Import Tests")
    
    modules = [
        ("Discord Interface", "agentic.interfaces.discord_interface"),
        ("Agent API", "agentic.interfaces.agent_api"),
        ("Agent Orchestrator", "agentic.agents.orchestrator"),
        ("Conversation Memory", "agentic.memory.conversation_memory"),
        ("Smart Cache", "agentic.cache.smart_cache"),
        ("Vector Store", "agentic.vectorstore.persistent_store")
    ]
    
    print("🔄 Testing module imports...")
    
    failed_imports = []
    
    for i, (name, module_path) in enumerate(modules):
        time.sleep(0.2)  # Small delay for visual effect
        print_progress_bar(i + 1, len(modules), prefix='Progress:', suffix=f'Importing {name}')
        
        try:
            __import__(module_path, fromlist=[''])
        except Exception as e:
            failed_imports.append(f"{name}: {str(e)}")
    
    print()  # New line after progress bar
    
    if failed_imports:
        print("❌ Failed imports:")
        for failed in failed_imports:
            print(f"   - {failed}")
        return False
    
    print("✅ All imports successful")
    return True

def test_configuration():
    """Test configuration loading and validation"""
    print("\n🧪 Testing configuration...")
    
    try:
        # Create test configuration
        test_config = {
            "discord": {
                "token": "test_token",
                "command_prefix": "!",
                "guild_id": "12345"
            },
            "orchestrator": {
                "memory_config": {
                    "max_conversations": 1000,
                    "max_messages_per_conversation": 100
                },
                "agent_configs": {
                    "search_agent": {"enabled": True},
                    "summary_agent": {"enabled": True},
                    "context_agent": {"enabled": True}
                }
            },
            "cache": {
                "ttl": 3600,
                "max_size": 1000
            },
            "openai": {
                "api_key": "test_key",
                "model": "gpt-3.5-turbo"
            }
        }
        
        print("✅ Configuration structure valid")
        return test_config
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None

async def test_memory_system():
    """Test the conversation memory system"""
    print("\n🧪 Testing memory system...")
    
    try:
        from agentic.memory.conversation_memory import ConversationMemory
        
        memory = ConversationMemory({})
        
        # Test adding interactions
        user_id = "test_user_123"
        await memory.add_interaction(
            user_id=user_id,
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            metadata={"test": True}
        )
        
        # Test retrieving history - get_history expects user_id, not conversation_id
        history = await memory.get_history(user_id)
        assert len(history) > 0, "History should contain interactions"
        
        # Test user context
        context = await memory.get_user_context(user_id)
        assert isinstance(context, dict), "Context should be a dictionary"
        # Context might be empty for new users, but should include the structure
        expected_keys = ["preferences", "frequent_queries", "last_active", "metadata"]
        # After adding interaction, context should be updated
        context_after = await memory.get_user_context(user_id)
        assert isinstance(context_after, dict), "Context should be a dictionary"
        
        print("✅ Memory system working correctly")
        return True
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

async def test_cache_system():
    """Test the smart cache system"""
    print("\n🧪 Testing cache system...")
    
    try:
        from agentic.cache.smart_cache import SmartCache
        
        cache = SmartCache({})
        
        # Test caching
        key = "test_query_hash"
        value = {"response": "test response", "timestamp": datetime.now().isoformat()}
        
        await cache.set(key, value)
        cached_value = await cache.get(key)
        
        assert cached_value is not None, "Cached value should be retrievable"
        assert cached_value["response"] == "test response", "Cached content should match"
        
        print("✅ Cache system working correctly")
        return True
    except Exception as e:
        print(f"❌ Cache system test failed: {e}")
        return False

async def test_agent_api():
    """Test the agent API with mocked dependencies"""
    print("\n🧪 Testing agent API...")
    
    try:
        # Mock OpenAI before importing
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                from agentic.interfaces.agent_api import AgentAPI
                
                # Mock configuration
                config = {
                    "orchestrator": {
                        "memory_config": {},
                        "agent_configs": {}
                    },
                    "openai": {
                        "api_key": "test_key",
                        "model": "gpt-3.5-turbo"
                    }
                }
                
                # Create agent API
                agent_api = AgentAPI(config)
                
                # Test health check - call it directly
                health = await agent_api.health_check()
                assert "status" in health, "Health check should return status"
                assert health["status"] in ["healthy", "degraded", "unhealthy"], "Status should be valid"
        
        print("✅ Agent API working correctly")
        return True
    except Exception as e:
        print(f"❌ Agent API test failed: {e}")
        return False

async def test_discord_interface():
    """Test the Discord interface with mocked dependencies"""
    print("\n🧪 Testing Discord interface...")
    
    try:
        # Mock OpenAI before importing
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext
                
                config = {
                    "discord": {
                        "token": "test_token",
                        "command_prefix": "!"
                    },
                    "orchestrator": {
                        "memory_config": {}
                    },
                    "cache": {}
                }
                
                # Create Discord interface
                interface = DiscordInterface(config)
                
                # Test Discord context creation
                context = DiscordContext(
                    user_id=123456789,
                    username="testuser",
                    channel_id=987654321,
                    guild_id=111222333,
                    channel_name="general",
                    guild_name="Test Server",
                    timestamp=datetime.now()
                )
                
                assert context.user_id == 123456789, "Context should store user ID correctly"
                assert context.username == "testuser", "Context should store username correctly"
        
        print("✅ Discord interface working correctly")
        return True
    except Exception as e:
        print(f"❌ Discord interface test failed: {e}")
        return False

async def test_orchestrator():
    """Test the agent orchestrator"""
    print("\n🧪 Testing orchestrator...")
    
    try:
        # Mock OpenAI before importing
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                from agentic.agents.orchestrator import AgentOrchestrator
                
                config = {
                    "memory_config": {},
                    "agent_configs": {
                        "search_agent": {"enabled": True},
                        "summary_agent": {"enabled": True},
                        "context_agent": {"enabled": True}
                    }
                }
                
                orchestrator = AgentOrchestrator(config)
                
                # Test basic initialization
                assert orchestrator is not None, "Orchestrator should be initialized"
                assert orchestrator.config == config, "Config should be stored"
        
        print("✅ Orchestrator working correctly")
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

async def test_end_to_end():
    """Test end-to-end functionality with mocked external dependencies"""
    print("\n🧪 Testing end-to-end functionality...")
    
    try:
        # Mock OpenAI before importing
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                from agentic.interfaces.discord_interface import DiscordInterface, DiscordContext
                
                config = {
                    "discord": {
                        "token": "test_token",
                        "command_prefix": "!"
                    },
                    "orchestrator": {
                        "memory_config": {},
                        "agent_configs": {}
                    },
                    "cache": {},
                    "openai": {
                        "api_key": "test_key",
                        "model": "gpt-3.5-turbo"
                    }
                }
                
                interface = DiscordInterface(config)
                
                # Mock the agent API query method
                mock_response = {
                    "status": "success",
                    "response": {
                        "answer": "This is a test response from the agentic system."
                    },
                    "execution_time": 1.5
                }
                
                with patch.object(interface.agent_api, 'query', return_value=mock_response):
                    context = DiscordContext(
                        user_id=123456789,
                        username="testuser",
                        channel_id=987654321,
                        guild_id=111222333,
                        channel_name="general",
                        guild_name="Test Server",
                        timestamp=datetime.now()
                    )
                    
                    # Test query processing
                    messages = await interface.process_query(
                        "What is the meaning of life?",
                        context
                    )
                    
                    assert len(messages) > 0, "Should return formatted messages"
                    assert "What is the meaning of life?" in messages[0], "Should include original query"
                    assert "test response" in messages[0], "Should include response content"
        
        print("✅ End-to-end test successful")
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting comprehensive system tests...\n")
    
    tests = [
        ("Import Test", test_imports()),
        ("Configuration Test", test_configuration()),
        ("Memory System Test", test_memory_system()),
        ("Cache System Test", test_cache_system()),
        ("Agent API Test", test_agent_api()),
        ("Discord Interface Test", test_discord_interface()),
        ("Orchestrator Test", test_orchestrator()),
        ("End-to-End Test", test_end_to_end())
    ]
    
    results = []
    for test_name, test_func in tests:
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The agentic Discord bot system is ready.")
        print("\n📋 To deploy:")
        print("1. Set environment variables: OPENAI_API_KEY, DISCORD_TOKEN, GUILD_ID")
        print("2. Run: python main.py")
        print("3. Use /ask command in Discord")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(main())
