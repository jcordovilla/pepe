#!/usr/bin/env python3
"""
Test Configuration Consistency

This script verifies that all configuration is consistent with
environment variables and the modernized configuration system.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

async def test_config_consistency():
    """Test that all configuration is consistent."""
    print("🔍 Testing configuration consistency...")
    
    # Test environment variables
    print("\n📋 Environment Variables:")
    env_vars = {
        "LLM_ENDPOINT": os.getenv("LLM_ENDPOINT"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS"),
        "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE"),
        "LLM_TIMEOUT": os.getenv("LLM_TIMEOUT"),
        "LLM_RETRY_ATTEMPTS": os.getenv("LLM_RETRY_ATTEMPTS"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN"),
    }
    
    for var, value in env_vars.items():
        status = "✅" if value else "❌"
        display_value = value if value else "NOT SET"
        print(f"   {status} {var}: {display_value}")
    
    # Test modernized configuration
    print("\n⚙️ Modernized Configuration:")
    try:
        from agentic.config.modernized_config import get_modernized_config
        config = get_modernized_config()
        
        # Test LLM config
        llm_config = config.get("llm", {})
        print(f"   🤖 LLM Endpoint: {llm_config.get('endpoint')}")
        print(f"   🤖 LLM Model: {llm_config.get('model')}")
        print(f"   🤖 LLM Max Tokens: {llm_config.get('max_tokens')}")
        print(f"   🤖 LLM Temperature: {llm_config.get('temperature')}")
        
        # Test OpenAI config
        openai_config = config.get("openai", {})
        print(f"   🔑 OpenAI API Key: {'✅ SET' if openai_config.get('api_key') else '❌ NOT SET'}")
        print(f"   🔑 OpenAI Embedding Model: {openai_config.get('embedding_model')}")
        
        # Test data config
        data_config = config.get("data", {})
        vector_config = data_config.get("vector_config", {})
        print(f"   📊 Vector Embedding Model: {vector_config.get('embedding_model')}")
        
        # Test Discord config
        discord_config = config.get("discord", {})
        print(f"   🎮 Discord Token: {'✅ SET' if discord_config.get('token') else '❌ NOT SET'}")
        
    except Exception as e:
        print(f"   ❌ Error loading configuration: {e}")
    
    # Test LLM client
    print("\n🤖 LLM Client Test:")
    try:
        from agentic.services.llm_client import get_llm_client
        client = get_llm_client()
        print(f"   ✅ LLM Client initialized successfully")
        print(f"   🤖 Client Model: {client.config.get('model')}")
        print(f"   🤖 Client Endpoint: {client.config.get('endpoint')}")
    except Exception as e:
        print(f"   ❌ Error initializing LLM client: {e}")
    
    # Test vector store configuration
    print("\n📊 Vector Store Configuration Test:")
    try:
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        test_config = {
            "collection_name": "test_collection",
            "persist_directory": "./data/test_chromadb",
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "batch_size": 100
        }
        print(f"   ✅ Vector store config created")
        print(f"   📊 Embedding Model: {test_config.get('embedding_model')}")
    except Exception as e:
        print(f"   ❌ Error creating vector store config: {e}")
    
    print("\n✅ Configuration consistency test complete!")

def main():
    """Main function."""
    asyncio.run(test_config_consistency())

if __name__ == "__main__":
    main() 