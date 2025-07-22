#!/usr/bin/env python3
"""
Test script for agent registry
"""

import asyncio
import json
from agentic.config.modernized_config import get_modernized_config
from agentic.interfaces.agent_api import AgentAPI
from agentic.agents.base_agent import SubTask, AgentRole, TaskStatus

async def test_agent_registry():
    """Test the agent registry"""
    
    print("🔧 Testing Agent Registry...")
    
    config = get_modernized_config()
    api = AgentAPI(config)
    
    # Test what agents are registered
    print("\n📊 Registered Agents:")
    from agentic.agents.base_agent import agent_registry
    registered_agents = agent_registry.list_agents()
    for role in registered_agents:
        agent = agent_registry.get_agent(role)
        print(f"  {role.value}: {type(agent).__name__}")
    
    # Test capability_response task
    print("\n📊 Testing capability_response task:")
    capability_task = SubTask(
        id="test_capability",
        description="Generate information about bot capabilities",
        agent_role=AgentRole.ANALYZER,
        task_type="capability_response",
        parameters={
            "query": "capability inquiry",
            "response_type": "capability"
        },
        dependencies=[],
        status=TaskStatus.PENDING
    )
    
    capable_agent = agent_registry.find_capable_agent(capability_task)
    if capable_agent:
        print(f"  ✅ Found capable agent: {type(capable_agent).__name__}")
        print(f"  ✅ Agent role: {capable_agent.role.value}")
    else:
        print(f"  ❌ No capable agent found for capability_response task")
    
    # Test semantic_search task
    print("\n📊 Testing semantic_search task:")
    search_task = SubTask(
        id="test_search",
        description="Search for relevant messages",
        agent_role=AgentRole.SEARCHER,
        task_type="semantic_search",
        parameters={
            "query": "test query",
            "filters": {},
            "k": 10
        },
        dependencies=[],
        status=TaskStatus.PENDING
    )
    
    capable_agent = agent_registry.find_capable_agent(search_task)
    if capable_agent:
        print(f"  ✅ Found capable agent: {type(capable_agent).__name__}")
        print(f"  ✅ Agent role: {capable_agent.role.value}")
    else:
        print(f"  ❌ No capable agent found for semantic_search task")
    
    print("\n🎯 Agent registry test completed!")

if __name__ == "__main__":
    asyncio.run(test_agent_registry()) 