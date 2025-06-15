"""
Test query analysis with channel ID resolution.
"""

import os
import sys
import asyncio
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

# Set dummy OpenAI key for testing
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from agentic.reasoning.query_analyzer import QueryAnalyzer

async def test_query_analysis():
    print("Testing Query Analysis with Channel ID Resolution")
    print("=" * 60)
    
    # Initialize analyzer
    config = {
        "model": "gpt-4-turbo",
        "chromadb_path": "./data/chromadb/chroma.sqlite3",
        "llm_complexity_threshold": 0.85,
    }
    analyzer = QueryAnalyzer(config)
    
    # Test queries
    test_queries = [
        "list the last 5 messages in agent-ops channel",
        "find messages in #🦾agent-ops about testing",
        "show me discussions in agent ops channel",
        "search for bugs in the agentops channel",
        "what was discussed in netarch-agents last week",
        "find messages in non-existent-channel"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        try:
            analysis = await analyzer.analyze(query)
            
            print(f"Intent: {analysis.get('intent', 'unknown')}")
            print(f"Complexity: {analysis.get('complexity', 0):.2f}")
            
            entities = analysis.get('entities', [])
            print(f"Entities found: {len(entities)}")
            
            for entity in entities:
                if entity['type'] == 'channel':
                    channel_id = entity.get('channel_id')
                    if channel_id:
                        print(f"  ✅ Channel: '{entity['value']}' -> ID: {channel_id} (confidence: {entity['confidence']:.2f})")
                    else:
                        print(f"  ❌ Channel: '{entity['value']}' -> Not resolved (confidence: {entity['confidence']:.2f})")
                else:
                    print(f"  {entity['type']}: '{entity['value']}' (confidence: {entity['confidence']:.2f})")
                    
        except Exception as e:
            print(f"Error analyzing query: {e}")

if __name__ == "__main__":
    asyncio.run(test_query_analysis())
