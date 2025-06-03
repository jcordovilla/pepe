#!/usr/bin/env python3
"""
Test script to verify the bot can search and return meaningful results
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agentic.interfaces.agent_api import AgentAPI

async def test_search_functionality():
    """Test the bot's search functionality with various queries"""
    
    print("🔍 Testing Discord Bot Search Functionality")
    print("=" * 50)
    
    try:
        # Initialize the AgentAPI
        print("📝 Initializing AgentAPI...")
        agent_api = AgentAPI()
        
        # Test queries that should find results in the Discord messages
        test_queries = [
            "machine learning",
            "AI tools and resources",
            "LLM models",
            "Python programming",
            "data science",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: Searching for '{query}'")
            print("-" * 40)
            
            try:
                response = await agent_api.process_query(
                    query=query,
                    user_id="test_user",
                    conversation_id="test_conversation"
                )
                
                print(f"📋 Response Type: {type(response)}")
                if isinstance(response, dict):
                    print(f"📄 Response Keys: {list(response.keys())}")
                    if response.get('answer'):
                        print(f"✅ Answer Length: {len(response['answer'])} characters")
                        print(f"📝 Answer Preview: {response['answer'][:200]}...")
                    else:
                        print("❌ No answer in response")
                        print(f"📄 Full Response: {response}")
                else:
                    print(f"📄 Response: {response}")
                
            except Exception as e:
                print(f"❌ Error processing query '{query}': {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Search functionality test completed")
        
    except Exception as e:
        print(f"❌ Failed to initialize AgentAPI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_functionality())
