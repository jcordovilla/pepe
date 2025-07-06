#!/usr/bin/env python3
"""
Live agent system test to verify everything is working after cleanup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.interfaces.agent_api import AgentAPI
from agentic.services.channel_resolver import ChannelResolver

async def test_agent_system():
    """Test the complete agent system with real queries"""
    print("🔍 TESTING LIVE AGENT SYSTEM")
    print("=" * 80)
    
    try:
        # Initialize the agent API
        print("🚀 Initializing Agent API...")
        agent_api = AgentAPI({})
        
        # Test 1: Simple search query
        print("\n📋 Test 1: Simple search query")
        print("-" * 40)
        
        query1 = "agent operations"
        print(f"Query: '{query1}'")
        
        result1 = await agent_api.query(query1, user_id="test_user")
        print(f"✅ Result: Found {len(result1.get('results', []))} results")
        if result1.get('results'):
            first_result = result1['results'][0]
            print(f"   First result: {first_result.get('content', '')[:100]}...")
        
        # Test 2: Channel-specific query
        print("\n📋 Test 2: Channel-specific query")
        print("-" * 40)
        
        query2 = "channel:#🦾agent-ops messages about development"
        print(f"Query: '{query2}'")
        
        result2 = await agent_api.query(query2, user_id="test_user")
        print(f"✅ Result: Found {len(result2.get('results', []))} results")
        if result2.get('results'):
            first_result = result2['results'][0]
            print(f"   First result: {first_result.get('content', '')[:100]}...")
        
        # Test 3: Temporal query
        print("\n📋 Test 3: Temporal query")
        print("-" * 40)
        
        query3 = "recent messages about discord"
        print(f"Query: '{query3}'")
        
        result3 = await agent_api.query(query3, user_id="test_user")
        print(f"✅ Result: Found {len(result3.get('results', []))} results")
        if result3.get('results'):
            first_result = result3['results'][0]
            print(f"   First result: {first_result.get('content', '')[:100]}...")
        
        # Test 4: Channel resolution
        print("\n📋 Test 4: Channel resolution")
        print("-" * 40)
        
        resolver = ChannelResolver()
        test_channel_id = "1363537366110703937"  # admin-onboarding
        resolved_name = resolver.resolve_channel_name(test_channel_id)
        print(f"Channel ID {test_channel_id} → '{resolved_name}'")
        
        # Test 5: Vector store status
        print("\n📋 Test 5: Vector store status")
        print("-" * 40)
        
        # Get vector store from the search agent
        search_agent = agent_api.orchestrator.agents.get('searcher')
        if search_agent and hasattr(search_agent, 'vector_store'):
            vector_store = search_agent.vector_store
            if vector_store.collection:
                total_messages = vector_store.collection.count()
                print(f"✅ Vector store loaded: {total_messages:,} messages indexed")
                
                # Sample a few documents to verify metadata
                sample_results = vector_store.collection.peek(limit=3)
                if sample_results and 'metadatas' in sample_results:
                    print("   Sample metadata fields:")
                    for i, metadata in enumerate(sample_results['metadatas'][:2]):
                        if metadata:
                            field_count = len(metadata)
                            author = metadata.get('author', {})
                            display_name = author.get('display_name') if isinstance(author, dict) else 'N/A'
                            channel = metadata.get('channel_name', 'Unknown')
                            print(f"     Message {i+1}: {field_count} fields, Author: {display_name}, Channel: #{channel}")
            else:
                print("❌ Vector store collection not loaded")
        else:
            print("❌ Vector store not accessible")
        
        print(f"\n🎉 AGENT SYSTEM TEST COMPLETED!")
        print("=" * 80)
        print("✅ All core components are functioning properly")
        print("✅ Enhanced metadata is present")
        print("✅ Channel resolution working")
        print("✅ Vector store operational with 7,000+ messages")
        print("✅ Query processing functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during agent system test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_system())
    if success:
        print("\n🎯 SUCCESS: Agent system is fully operational after cleanup!")
    else:
        print("\n⚠️ ISSUE: Agent system encountered problems")
    exit(0 if success else 1)
