#!/usr/bin/env python3
"""
Simple test script to verify local AI setup is working.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_client import get_ai_client
from core.config import get_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_local_ai_setup():
    """Test that local AI components are working."""
    print("🔍 Testing Local AI Setup...")
    
    try:
        # Initialize AI client
        ai_client = get_ai_client()
        config = get_config()
        
        print(f"✅ AI Client initialized")
        print(f"📝 Chat Model: {config.models.chat_model}")
        print(f"🔤 Embedding Model: {config.models.embedding_model}")
        print(f"🔗 Ollama URL: {config.models.ollama_base_url}")
        
        # Test health check
        print("\n🔍 Running health check...")
        health = ai_client.health_check()
        print(f"Health Status: {health}")
        
        if not health['overall']:
            print("❌ Health check failed. Please ensure:")
            print("  1. Ollama is running: ollama serve")
            print("  2. Required model is available: ollama pull llama2:latest")
            return False
        
        # Test embeddings
        print("\n🔤 Testing embeddings...")
        test_text = "This is a test message for embeddings"
        embeddings = ai_client.create_embeddings(test_text)
        print(f"✅ Embeddings created: shape {embeddings.shape}")
        
        # Test chat completion
        print("\n💬 Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Local AI is working!' and nothing else."}
        ]
        response = ai_client.chat_completion(messages, temperature=0.0, max_tokens=50)
        print(f"✅ Chat response: {response}")
        
        print("\n🎉 All tests passed! Local AI setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_local_ai_setup()
    sys.exit(0 if success else 1)
