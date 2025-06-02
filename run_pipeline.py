#!/usr/bin/env python3
"""
Quick script to run the Discord message database update pipeline.
"""

import asyncio
import logging
from agentic.interfaces.agent_api import AgentAPI

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    """Run the pipeline update"""
    print("🚀 Starting Discord message database update...")
    
    # Initialize the API
    config = {
        "orchestrator": {},
        "vectorstore": {"persist_directory": "data/vectorstore"},
        "memory": {"db_path": "data/conversation_memory.db"},
        "pipeline": {"base_path": "."},
        "analytics": {"db_path": "data/analytics.db"}
    }
    
    agent_api = AgentAPI(config)
    
    try:
        # Check current status
        status = agent_api.get_pipeline_status()
        print(f"📊 Pipeline Status: {'Running' if status.get('is_running') else 'Ready'}")
        
        # Run full pipeline
        print("🔄 Running full pipeline (fetch → embed → detect → sync)...")
        result = await agent_api.run_pipeline("manual_update")
        
        if result.get("success"):
            print("✅ Pipeline completed successfully!")
            if "stats" in result:
                stats = result["stats"]
                print(f"📈 Results: {stats.get('total_messages', 'N/A')} messages, {stats.get('total_resources', 'N/A')} resources")
        else:
            print(f"❌ Pipeline failed: {result.get('error')}")
            if "failed_step" in result:
                print(f"💥 Failed at step: {result['failed_step']}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await agent_api.close()

if __name__ == "__main__":
    asyncio.run(main())
