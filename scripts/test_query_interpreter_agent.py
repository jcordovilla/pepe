#!/usr/bin/env python3
"""
Test Script for QueryInterpreterAgent

Tests the new LLM-powered query interpreter to ensure it correctly:
1. Interprets user queries
2. Extracts intent and entities
3. Suggests appropriate subtasks
4. Provides confidence scores and rationale
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.agents.query_interpreter_agent import QueryInterpreterAgent
from agentic.agents.base_agent import AgentState


async def test_query_interpreter():
    """Test the QueryInterpreterAgent with various queries"""
    
    print("🧪 Testing QueryInterpreterAgent")
    print("=" * 50)
    
    # Initialize the agent
    config = {
        "model": "llama-3.1-8b-instruct",
        "max_tokens": 2048,
        "temperature": 0.1,
        "cache": {"enabled": True, "ttl": 3600},
        "cache_ttl": 3600
    }
    
    agent = QueryInterpreterAgent(config)
    
    # Test queries
    test_queries = [
        "summarise the messages from #general in the past week",
        "what can you do?",
        "search for messages about Python programming",
        "find messages with reactions in #help",
        "analyze the discussion trends in the last month",
        "extract skills mentioned in the tech channel",
        "show me the latest messages from @john",
        "what resources were shared recently?",
        "create a weekly digest of activity",
        "search for messages containing 'bug' or 'error'"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 40)
        
        # Create test state
        state: AgentState = {
            "messages": [{"role": "user", "content": query}],
            "search_results": [],
            "conversation_history": [],
            "user_context": {
                "user_id": "test_user",
                "query": query,
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
            # Process the query
            start_time = datetime.utcnow()
            result_state = await agent.process(state)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract interpretation results
            interpretation = result_state.get("query_interpretation", {})
            
            print(f"✅ Intent: {interpretation.get('intent', 'unknown')}")
            print(f"🎯 Confidence: {interpretation.get('confidence', 0.0):.2f}")
            print(f"💭 Rationale: {interpretation.get('rationale', 'N/A')}")
            
            # Show entities
            entities = interpretation.get("entities", [])
            if entities:
                print(f"🏷️  Entities ({len(entities)}):")
                for entity in entities:
                    print(f"   - {entity.get('type')}: {entity.get('value')} (confidence: {entity.get('confidence', 0.0):.2f})")
            else:
                print("🏷️  Entities: None")
            
            # Show suggested subtasks
            subtasks = interpretation.get("subtasks", [])
            if subtasks:
                print(f"📋 Subtasks ({len(subtasks)}):")
                for j, subtask in enumerate(subtasks, 1):
                    print(f"   {j}. {subtask.get('task_type')}: {subtask.get('description')}")
                    params = subtask.get("parameters", {})
                    if params:
                        print(f"      Parameters: {json.dumps(params, indent=6)}")
                    deps = subtask.get("dependencies", [])
                    if deps:
                        print(f"      Dependencies: {deps}")
            else:
                print("📋 Subtasks: None")
            
            print(f"⏱️  Processing time: {duration:.3f}s")
            
            # Store results
            results.append({
                "query": query,
                "interpretation": interpretation,
                "duration": duration,
                "success": True
            })
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        avg_duration = sum(r["duration"] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average processing time: {avg_duration:.3f}s")
        
        # Intent distribution
        intents = {}
        for result in successful_tests:
            intent = result["interpretation"].get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1
        
        print(f"🎯 Intent distribution:")
        for intent, count in intents.items():
            print(f"   - {intent}: {count}")
        
        # Confidence analysis
        confidences = [r["interpretation"].get("confidence", 0.0) for r in successful_tests]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"🎯 Average confidence: {avg_confidence:.2f}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for result in failed_tests:
            print(f"   - {result['query']}: {result['error']}")
    
    # Save detailed results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_query_interpreter_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "query_interpreter_agent",
            "results": results,
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "avg_duration": avg_duration if successful_tests else 0,
                "avg_confidence": avg_confidence if successful_tests else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return len(successful_tests) == len(results)


async def test_agent_registration():
    """Test that the QueryInterpreterAgent can be registered and found"""
    
    print("\n🔧 Testing Agent Registration")
    print("=" * 30)
    
    from agentic.agents.base_agent import agent_registry, SubTask, AgentRole
    
    # Create and register the agent
    config = {"model": "llama-3.1-8b-instruct"}
    agent = QueryInterpreterAgent(config)
    agent_registry.register_agent(agent)
    
    # Test task handling
    test_tasks = [
        SubTask(
            id="test_1",
            description="Interpret user query",
            agent_role=AgentRole.ANALYZER,
            task_type="interpret_query",
            parameters={"query": "test query"},
            dependencies=[]
        ),
        SubTask(
            id="test_2", 
            description="Analyze query intent",
            agent_role=AgentRole.ANALYZER,
            task_type="analyze_query",
            parameters={"query": "test query"},
            dependencies=[]
        ),
        SubTask(
            id="test_3",
            description="Extract intent from query", 
            agent_role=AgentRole.ANALYZER,
            task_type="extract_intent",
            parameters={"query": "test query"},
            dependencies=[]
        )
    ]
    
    for task in test_tasks:
        capable_agent = agent_registry.find_capable_agent(task)
        if capable_agent and isinstance(capable_agent, QueryInterpreterAgent):
            print(f"✅ Task '{task.task_type}' can be handled by QueryInterpreterAgent")
        else:
            print(f"❌ Task '{task.task_type}' cannot be handled by QueryInterpreterAgent")
    
    # List registered agents
    agents = agent_registry.list_agents()
    print(f"📋 Registered agent roles: {[role.value for role in agents]}")


if __name__ == "__main__":
    async def main():
        """Run all tests"""
        print("🚀 Starting QueryInterpreterAgent Tests")
        print("=" * 60)
        
        # Test agent registration
        await test_agent_registration()
        
        # Test query interpretation
        success = await test_query_interpreter()
        
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests failed!")
            sys.exit(1)
    
    asyncio.run(main()) 