#!/usr/bin/env python3
"""
Test LLM Consistency

Verifies that all modules use the same Llama model consistently.
Tests the unified LLM client and all agents that use LLM calls.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.services.llm_client import get_llm_client, UnifiedLLMClient
from agentic.agents.query_interpreter_agent import QueryInterpreterAgent
from agentic.agents.analysis_agent import AnalysisAgent
from agentic.memory.enhanced_memory import EnhancedConversationMemory
from agentic.agents.base_agent import AgentState


async def test_llm_client():
    """Test the unified LLM client"""
    print("🧪 Testing Unified LLM Client")
    print("=" * 40)
    
    client = get_llm_client()
    
    # Test health check
    print("📊 Health Check:")
    health = await client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Primary Model: {health['primary_model']}")
    print(f"   Fallback Model: {health['fallback_model']}")
    print(f"   Endpoint: {health['endpoint']}")
    
    if health['status'] == 'healthy':
        print(f"   Test Response: {health['test_response']}")
    else:
        print(f"   Error: {health['error']}")
    
    # Test available models
    print("\n📋 Available Models:")
    models = await client.get_available_models()
    for model in models:
        print(f"   - {model['name']} ({model['model']})")
        if 'details' in model:
            details = model['details']
            print(f"     Size: {details.get('parameter_size', 'Unknown')}")
            print(f"     Quantization: {details.get('quantization_level', 'Unknown')}")
    
    # Test basic generation
    print("\n💬 Testing Basic Generation:")
    try:
        response = await client.generate(
            prompt="What is 2+2? Please respond with just the number.",
            max_tokens=10,
            temperature=0.0
        )
        print(f"   Response: {response.strip()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test JSON generation
    print("\n📄 Testing JSON Generation:")
    try:
        response = await client.generate_json(
            prompt="Create a simple JSON object with name and age fields.",
            max_tokens=50,
            temperature=0.1
        )
        print(f"   Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return health['status'] == 'healthy'


async def test_query_interpreter_agent():
    """Test QueryInterpreterAgent LLM usage"""
    print("\n🧠 Testing QueryInterpreterAgent")
    print("=" * 40)
    
    config = {
        "model": "llama3.1:8b",
        "max_tokens": 2048,
        "temperature": 0.1
    }
    
    agent = QueryInterpreterAgent(config)
    
    # Create test state
    state: AgentState = {
        "messages": [{"role": "user", "content": "summarise messages from #general"}],
        "search_results": [],
        "conversation_history": [],
        "user_context": {
            "user_id": "test_user",
            "query": "summarise messages from #general",
            "platform": "discord",
            "timestamp": datetime.utcnow().isoformat()
        },
        "task_plan": None,
        "current_step": 0,
        "metadata": {
            "start_time": datetime.utcnow().timestamp(),
            "version": "1.0.0"
        }
    }
    
    try:
        result_state = await agent.process(state)
        interpretation = result_state.get("query_interpretation", {})
        
        print(f"✅ Intent: {interpretation.get('intent', 'unknown')}")
        print(f"🎯 Confidence: {interpretation.get('confidence', 0.0):.2f}")
        print(f"📋 Subtasks: {len(interpretation.get('subtasks', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_analysis_agent():
    """Test AnalysisAgent LLM usage"""
    print("\n📊 Testing AnalysisAgent")
    print("=" * 40)
    
    config = {
        "summary_length": 200,
        "analysis_depth": "comprehensive"
    }
    
    agent = AnalysisAgent(config)
    
    # Create test subtask
    from agentic.agents.base_agent import SubTask, AgentRole
    subtask = SubTask(
        id="test_summary",
        description="Test summarization",
        agent_role=AgentRole.ANALYZER,
        task_type="summarize",
        parameters={
            "content_source": "search_results",
            "summary_type": "overview",
            "focus_areas": ["key topics"]
        },
        dependencies=[]
    )
    
    # Create test state
    state: AgentState = {
        "messages": [{"role": "user", "content": "summarize this"}],
        "search_results": [
            {
                "content": "This is a test message about Python programming. The discussion covered various topics including web development and data science.",
                "metadata": {"author": "test_user", "timestamp": "2024-01-01T00:00:00Z"}
            }
        ],
        "conversation_history": [],
        "user_context": {"user_id": "test_user"},
        "task_plan": None,
        "current_step": 0,
        "metadata": {}
    }
    
    try:
        result = await agent._summarize_content(subtask, state)
        summary = result.get("summary", "")
        
        print(f"✅ Summary generated: {len(summary)} characters")
        print(f"📝 Preview: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_enhanced_memory():
    """Test EnhancedConversationMemory LLM usage"""
    print("\n🧠 Testing EnhancedConversationMemory")
    print("=" * 40)
    
    config = {
        "max_active_memory": 50,
        "summary_threshold": 20
    }
    
    memory = EnhancedConversationMemory(config)
    
    # Test conversation summarization
    conversations = [
        {
            "query": "What is Python?",
            "response": "Python is a programming language.",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        {
            "query": "How do I install Python?",
            "response": "You can download Python from python.org",
            "timestamp": "2024-01-01T00:01:00Z"
        }
    ]
    
    try:
        summary = await memory.smart_summarize_conversation(conversations)
        
        print(f"✅ Summary generated: {len(summary)} characters")
        print(f"📝 Preview: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_model_consistency():
    """Test that all modules use the same model"""
    print("\n🔍 Testing Model Consistency")
    print("=" * 40)
    
    # Get the unified client
    client = get_llm_client()
    
    # Check that all modules reference the same client
    print(f"✅ Unified Client Model: {client.model}")
    print(f"✅ Unified Client Endpoint: {client.endpoint}")
    
    # Test that different modules use the same configuration
    config = get_llm_client().config
    
    print(f"✅ Configuration Model: {config.get('model')}")
    print(f"✅ Configuration Endpoint: {config.get('endpoint')}")
    
    # Verify consistency
    if client.model == config.get('model'):
        print("✅ Model consistency verified")
        return True
    else:
        print("❌ Model inconsistency detected")
        return False


async def main():
    """Run all LLM consistency tests"""
    print("🚀 Starting LLM Consistency Tests")
    print("=" * 60)
    
    results = {}
    
    # Test LLM client
    results['llm_client'] = await test_llm_client()
    
    # Test model consistency
    results['model_consistency'] = await test_model_consistency()
    
    # Test QueryInterpreterAgent
    results['query_interpreter'] = await test_query_interpreter_agent()
    
    # Test AnalysisAgent
    results['analysis_agent'] = await test_analysis_agent()
    
    # Test EnhancedMemory
    results['enhanced_memory'] = await test_enhanced_memory()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_llm_consistency_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "llm_consistency",
            "results": results,
            "summary": {
                "all_passed": all_passed,
                "total_tests": len(results),
                "passed_tests": sum(results.values())
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 