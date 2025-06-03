"""
Test end-to-end channel ID filtering with the user's original failing query.
"""

import os
import sys
import asyncio
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

# Set dummy OpenAI key for testing
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from agentic.interfaces.agent_api import AgentAPI
from agentic.reasoning.query_analyzer import QueryAnalyzer
from agentic.reasoning.task_planner import TaskPlanner
from agentic.vectorstore.persistent_store import PersistentVectorStore

async def test_end_to_end_channel_filtering():
    print("Testing End-to-End Channel ID Filtering")
    print("=" * 50)
    
    # Test the user's original failing query
    test_query = "list the last 5 messages in agent-ops channel"
    
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    # 1. Test Query Analysis
    print("1. Query Analysis:")
    config = {
        "model": "gpt-4-turbo",
        "chromadb_path": "./data/chromadb/chroma.sqlite3"
    }
    analyzer = QueryAnalyzer(config)
    analysis = await analyzer.analyze(test_query)
    
    print(f"   Intent: {analysis.get('intent')}")
    entities = analysis.get('entities', [])
    channel_entity = None
    for entity in entities:
        if entity['type'] == 'channel':
            channel_entity = entity
            channel_id = entity.get('channel_id')
            print(f"   Channel: '{entity['value']}' -> ID: {channel_id}")
            break
    
    if not channel_entity or not channel_entity.get('channel_id'):
        print("   ❌ Channel ID not resolved")
        return
    
    # 2. Test Task Planning
    print("\n2. Task Planning:")
    planner = TaskPlanner(config)
    plan = await planner.create_plan(test_query, analysis, {})
    
    if plan and plan.subtasks:
        subtask = plan.subtasks[0]
        filters = subtask.parameters.get('filters', {})
        print(f"   Filters: {filters}")
        
        if 'channel_id' in filters:
            print(f"   ✅ Using channel_id filter: {filters['channel_id']}")
        else:
            print(f"   ❌ No channel_id filter found")
    
    # 3. Test Vector Store Search
    print("\n3. Vector Store Search:")
    try:
        vector_store = PersistentVectorStore({
            "persist_directory": "./data/chromadb",
            "collection_name": "discord_messages"
        })
        
        # Test with channel_id filter
        channel_id = channel_entity.get('channel_id')
        results = await vector_store.similarity_search(
            query="messages",
            k=5,
            filters={"channel_id": channel_id}
        )
        
        print(f"   Found {len(results)} results with channel_id filter")
        
        # Verify all results are from the correct channel
        correct_channel_count = 0
        for result in results:
            result_channel_id = result.get('channel_id')
            result_channel_name = result.get('channel_name', 'Unknown')
            if result_channel_id == channel_id:
                correct_channel_count += 1
            print(f"   - Message from: {result_channel_name} (ID: {result_channel_id})")
        
        if correct_channel_count == len(results):
            print(f"   ✅ All {len(results)} messages are from the correct channel")
        else:
            print(f"   ❌ Only {correct_channel_count}/{len(results)} messages from correct channel")
            
    except Exception as e:
        print(f"   Error testing vector store: {e}")

if __name__ == "__main__":
    asyncio.run(test_end_to_end_channel_filtering())
