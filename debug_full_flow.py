#!/usr/bin/env python3
"""
Debug script for the full flow
"""

import asyncio
import json
from agentic.config.modernized_config import get_modernized_config
from agentic.interfaces.agent_api import AgentAPI

async def debug_full_flow():
    """Debug the full flow from query to response"""
    
    print("🔧 Debugging Full Flow...")
    
    config = get_modernized_config()
    api = AgentAPI(config)
    
    query = "What is this bot capable of?"
    print(f"\n📝 Query: {query}")
    
    try:
        # Test the full flow
        print("\n🔍 Testing full API flow...")
        result = await api.query(
            query=query,
            user_id="test_user",
            context={"platform": "test"}
        )
        
        print(f"✅ Success: {result.get('success', False)}")
        print(f"✅ Response: {result.get('response', 'No response')[:500]}...")
        
        if not result.get('success', False):
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Check if there are any errors in the result
        if 'errors' in result:
            print(f"❌ Errors: {result['errors']}")
        
        # Check metadata
        if 'metadata' in result:
            print(f"📊 Metadata: {result['metadata']}")
        
    except Exception as e:
        print(f"💥 Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Full flow debug completed!")

if __name__ == "__main__":
    asyncio.run(debug_full_flow()) 