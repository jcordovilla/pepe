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
            "fast_model": os.getenv("LLM_FAST_MODEL", "phi3:mini"),  # Fast model for resource detection
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                                  "timeout": int(os.getenv("LLM_TIMEOUT", "120")),  # Increased timeout for 70B model
            "retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            "fallback_model": os.getenv("LLM_FALLBACK_MODEL", "llama2:latest")  # Fallback if primary model fails
        },
        
        # Modern unified data layer
        "data": {
            "vector_config": {
                "persist_directory": "./data/chromadb",
                "collection_name": "discord_messages",
                "embedding_model": os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
                "embedding_type": "sentence_transformers",  # Only using sentence-transformers
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
        
        # OpenAI configuration removed - using local models only
        
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
        },
        
        # Dynamic K-value configuration for query-appropriate result sizing
        "k_value_config": {
            # Base k values for different query types
            "base_values": {
                "simple_search": 10,
                "detailed_search": 25,
                "comprehensive_search": 50,
                "analysis": 75,
                "skill_experience": 100,  # Higher k for skill/experience queries
                "trend_analysis": 100,
                "digest": 150,
                "cross_server_analysis": 200
            },
            
            # Multipliers based on query characteristics
            "multipliers": {
                "time_range": {
                    "today": 1.0,
                    "yesterday": 1.2,
                    "this_week": 1.5,
                    "last_week": 1.8,
                    "this_month": 2.0,
                    "last_month": 2.5,
                    "all_time": 3.0
                },
                "scope": {
                    "single_channel": 1.0,
                    "multiple_channels": 1.5,
                    "cross_server": 2.0,
                    "all_channels": 2.5
                },
                "complexity": {
                    "simple": 1.0,
                    "moderate": 1.3,
                    "complex": 1.8,
                    "very_complex": 2.5
                }
            },
            
            # Query type patterns for automatic classification
            "query_patterns": {
                "comprehensive_analysis": [
                    "trending", "patterns", "overview", "summary", "digest", 
                    "analysis", "insights", "statistics", "metrics"
                ],
                "broad_search": [
                    "all", "everything", "comprehensive", "complete", "full",
                    "extensive", "thorough", "detailed"
                ],
                "user_analysis": [
                    "users", "people", "who", "active users", "engagement",
                    "participation", "contributors", "members"
                ],
                "skill_experience": [
                    "experience", "skills", "background", "expertise", "knowledge",
                    "professional", "certified", "worked", "career", "qualifications"
                ],
                "topic_analysis": [
                    "topics", "discussions", "conversations", "themes",
                    "subjects", "content types", "categories"
                ],
                "time_analysis": [
                    "recent", "history", "timeline", "evolution", "changes",
                    "growth", "development", "progress"
                ]
            },
            
            # Maximum k values for different contexts
            "max_values": {
                "search": 100,
                "analysis": 200,
                "digest": 300,
                "cross_server": 500
            },
            
            # Performance thresholds
            "performance": {
                "max_query_time": 30,  # seconds
                "enable_caching": True,
                "cache_ttl": 3600,  # 1 hour
                "batch_size": 50
            }
        }
    }
    
    return base_config

# Global configuration instance
config = get_modernized_config()
