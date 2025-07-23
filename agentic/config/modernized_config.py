"""
Modernized System Configuration
Enhanced from legacy patterns with unified architecture
"""

import os
from pathlib import Path
from typing import Dict, Any

def get_modernized_config() -> Dict[str, Any]:
    """
    Get modernized system configuration
    Preserves legacy-proven settings while adding modern features
    """
    
    base_config = {
        # Legacy-proven core settings
        "discord": {
            "token": os.getenv("DISCORD_TOKEN"),
            "page_size": 100,  # From legacy fetch_messages.py
            "rate_limit_delay": 1.0,  # From legacy patterns
            "max_retries": 3  # From legacy error handling
        },
        
        # Unified LLM configuration for all modules
        "llm": {
            "endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:11434/api/generate"),
            "model": os.getenv("LLM_MODEL", "llama3.1:8b"),  # Recommended: newer, better model
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
            "retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            "fallback_model": "llama2:latest"  # Fallback if primary model fails
        },
        
        # Modern unified data layer
        "data": {
            "vector_config": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages",
                "embedding_model": os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
                "embedding_type": os.getenv("EMBEDDING_TYPE", "sentence_transformers"),  # sentence_transformers or openai
                "batch_size": 100  # From legacy batch processing
            },
            "memory_config": {
                "database_url": "sqlite:///data/conversation_memory.db",
                "max_history": 50  # From legacy memory patterns
            },
            "cache_config": {
                "cache_dir": "./data/cache",
                "default_ttl": 3600,  # From legacy cache patterns
                "max_size_mb": 1000
            },
            "analytics_config": {
                "database_url": "sqlite:///data/analytics.db",
                "track_queries": True,  # From legacy analytics
                "track_performance": True
            }
        },
        
        # OpenAI configuration (fallback only - not used by default)
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "max_tokens": 4000,
            "temperature": 0.1  # From legacy proven settings
        },
        
        # Sentence Transformers configuration (for local embeddings)
        "sentence_transformers": {
            "model_name": os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
            "device": os.getenv("EMBEDDING_DEVICE", "cpu"),  # cpu or cuda
            "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
            "normalize_embeddings": True
        },
        
        # Legacy-proven processing settings
        "processing": {
            "batch_size": 100,  # From legacy batch_detect.py
            "max_retries": 3,   # From legacy error handling
            "rate_limit_delay": 1.0,  # From legacy patterns
            "enable_ai_classification": True,  # Modern enhancement
            "preserve_legacy_patterns": True
        },
        
        # Modern interface configuration  
        "interfaces": {
            "discord_enabled": True,
            "api_enabled": True,
            "api_port": 8000
        }
    }
    
    return base_config

# Global configuration instance
config = get_modernized_config()
