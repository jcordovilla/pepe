#!/usr/bin/env python3
"""
Test the exact flow that the Discord /vectorstats command uses
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agentic.interfaces.agent_api import AgentAPI

async def test_vectorstats_flow():
    """Test the exact flow used by the Discord /vectorstats command"""
    
    print("🔍 Testing Discord /vectorstats Command Flow")
    print("=" * 50)
    
    try:
        # Initialize AgentAPI (same as Discord interface does)
        print("📝 Initializing AgentAPI...")
        agent_api = AgentAPI()
        
        # Get vector store statistics (same call as Discord command)
        print("📊 Getting vector store statistics...")
        vector_stats = await agent_api.get_vector_stats()
        
        print(f"📋 Raw vector_stats: {vector_stats}")
        print()
        
        # Simulate the exact Discord command logic
        response = "📊 **Vector Store Statistics**\n\n"
        response += f"**Collection:** {vector_stats.get('collection_name', 'N/A')}\n"
        response += f"**Total Documents:** {vector_stats.get('total_documents', 0)}\n"
        response += f"**Embedding Model:** {vector_stats.get('embedding_model', 'N/A')}\n"
        response += f"**Last Updated:** {vector_stats.get('last_updated', 'N/A')}\n\n"
        
        # Add content statistics if available
        if "content_stats" in vector_stats:
            content_stats = vector_stats["content_stats"]
            response += "**Content Statistics:**\n"
            response += f"• Total Tokens: {content_stats.get('total_tokens', 0)}\n"
            response += f"• Average Length: {content_stats.get('avg_length', 0):.1f} tokens\n"
            response += f"• Longest Document: {content_stats.get('max_length', 0)} tokens\n\n"
            
            print("🔍 Content Stats Debug:")
            print(f"  📊 Raw content_stats: {content_stats}")
            print(f"  🔢 total_tokens value: {content_stats.get('total_tokens', 'NOT FOUND')}")
            print(f"  📏 avg_length value: {content_stats.get('avg_length', 'NOT FOUND')}")
            print(f"  📐 max_length value: {content_stats.get('max_length', 'NOT FOUND')}")
        else:
            response += "**Content Statistics:** Not available\n\n"
            print("❌ No content_stats found in vector_stats")
        
        # Add top channels if available
        if "top_channels" in vector_stats:
            response += "**Top Channels:**\n"
            for channel in vector_stats["top_channels"][:5]:
                response += f"• #{channel['name']}: {channel['count']} messages\n"
            response += "\n"
        
        # Add top authors if available  
        if "top_authors" in vector_stats:
            response += "**Top Authors:**\n"
            for author in vector_stats["top_authors"][:5]:
                response += f"• {author['username']}: {author['count']} messages\n"
        
        print("🎯 Final Discord Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        print(f"\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vectorstats_flow())
