#!/usr/bin/env python3
"""
Test the fixed Discord channel mention parsing.
"""

import asyncio
import os
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.reasoning.query_analyzer import QueryAnalyzer

async def test_channel_mention_parsing():
    """Test that Discord channel mentions are parsed correctly"""
    print("🔍 Testing Discord Channel Mention Parsing")
    print("=" * 60)
    
    try:
        # Initialize query analyzer
        config = {
            "model": "gpt-4-turbo",
            "chromadb_path": "./data/chromadb/chroma.sqlite3"
        }
        
        analyzer = QueryAnalyzer(config)
        
        # Test queries with Discord channel mentions
        test_queries = [
            "fetch me the last 5 messages from <#1365732945859444767>",
            "what discussions have taken place in <#1371647370911154228> ?",
            "show me content from <#1353448986408779877>",
            "search for AI in <#1353448986408779877>",
        ]
        
        print("🧪 Testing query parsing:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Parse the query
            analysis = await analyzer.analyze(query)
            
            print(f"   Intent: {analysis.get('intent', 'unknown')}")
            entities = analysis.get('entities', [])
            
            # Find channel entities
            channel_entities = [e for e in entities if e['type'] == 'channel']
            
            if channel_entities:
                for entity in channel_entities:
                    channel_id = entity.get('channel_id')
                    value = entity.get('value')
                    confidence = entity.get('confidence', 0)
                    
                    print(f"   ✅ Channel Entity Found:")
                    print(f"      Value: '{value}'")
                    print(f"      Channel ID: {channel_id}")
                    print(f"      Confidence: {confidence}")
                    
                    if channel_id and value != channel_id:
                        print(f"      🎉 Successfully resolved ID to name!")
                    elif channel_id:
                        print(f"      ⚠️ Has ID but resolution may need improvement")
                    else:
                        print(f"      ❌ No channel ID found")
            else:
                print(f"   ❌ No channel entities found")
        
        # Test the regex pattern directly
        print(f"\n🔧 Testing regex pattern directly:")
        
        channel_pattern = r"<#(\d+)>|(?:in|from)\s+(?:the\s+)?#?([\w-]+(?:\s+[\w-]+)*)\s+channel|#([\w🦾🤖🏛🗂❌💻📚🛠❓🌎🏘👋💠-]+)|(?:in|from)\s+([\w-]+-(?:ops|dev|agents|chat|help|support|resources))"
        
        test_strings = [
            "<#1365732945859444767>",
            "from <#1371647370911154228>",
            "in <#1353448986408779877> channel"
        ]
        
        for test_string in test_strings:
            matches = re.findall(channel_pattern, test_string)
            print(f"   '{test_string}' -> {matches}")
        
        print("\n✅ Channel mention parsing test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_channel_mention_parsing())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
