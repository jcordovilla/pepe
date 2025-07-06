#!/usr/bin/env python3
"""
Performance Optimizations Script

Applies immediate performance improvements:
1. Connection pooling optimizations
2. Embedding caching configuration 
3. Memory optimization settings
4. Batch processing improvements
5. Rate limiting configurations
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"⚡ {title}")
    print('=' * 60)

def optimize_vector_store_config():
    """Create optimized vector store configuration"""
    print_header("Vector Store Optimizations")
    
    config = {
        "collection_name": "discord_messages",
        "persist_directory": "./data/chromadb",
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "batch_size": 150,  # Increased for better performance
        "max_workers": 3,   # Parallel processing
        "cache": {
            "type": "hybrid",  # Memory + file cache
            "memory_ttl": 3600,
            "file_ttl": 86400,
            "max_memory_items": 2000,
            "max_file_size_mb": 500
        },
        "connection_pool": {
            "max_connections": 10,
            "keep_alive": True,
            "timeout": 30
        },
        "embedding_cache": {
            "enabled": True,
            "directory": "./data/cache/embeddings",
            "max_size_mb": 1000,
            "compression": True
        }
    }
    
    config_path = "data/optimized_vector_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Vector store config saved: {config_path}")
    return config

def optimize_memory_settings():
    """Configure memory optimization settings"""
    print_header("Memory Optimizations")
    
    config = {
        "conversation_memory": {
            "db_path": "data/conversation_memory.db",
            "max_history_length": 50,
            "context_window_hours": 24,
            "batch_size": 100,
            "connection_pool_size": 5,
            "cache_size": 1000,
            "enable_compression": True
        },
        "cache_settings": {
            "max_memory_usage_mb": 512,
            "cleanup_interval_seconds": 300,
            "auto_cleanup_enabled": True
        }
    }
    
    config_path = "data/optimized_memory_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Memory config saved: {config_path}")
    return config

def optimize_discord_settings():
    """Configure Discord performance settings"""
    print_header("Discord Interface Optimizations")
    
    config = {
        "discord": {
            "command_timeout": 15,  # Prevent Discord timeouts
            "response_chunk_size": 1900,  # Safe message size
            "max_search_results": 20,  # Limit results for performance
            "enable_typing_indicator": True,
            "defer_response": True,  # Always defer for long operations
            "batch_responses": True
        },
        "rate_limiting": {
            "messages_per_minute": 30,
            "queries_per_user_per_minute": 5,
            "cooldown_seconds": 2
        },
        "caching": {
            "cache_responses": True,
            "response_ttl_seconds": 300,
            "max_cached_responses": 1000
        }
    }
    
    config_path = "data/optimized_discord_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Discord config saved: {config_path}")
    return config

def optimize_analytics_settings():
    """Configure analytics performance settings"""
    print_header("Analytics Optimizations")
    
    config = {
        "analytics": {
            "async_recording": True,
            "batch_size": 50,
            "flush_interval_seconds": 30,
            "retention_days": 90,
            "enable_performance_tracking": True,
            "track_cache_hits": True
        },
        "monitoring": {
            "track_response_times": True,
            "track_memory_usage": True,
            "alert_thresholds": {
                "response_time_ms": 5000,
                "memory_usage_mb": 1000,
                "error_rate_percent": 5
            }
        }
    }
    
    config_path = "data/optimized_analytics_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Analytics config saved: {config_path}")
    return config

def create_performance_monitoring_config():
    """Create performance monitoring configuration"""
    print_header("Performance Monitoring Setup")
    
    config = {
        "monitoring": {
            "enabled": True,
            "interval_seconds": 60,
            "metrics": [
                "response_time",
                "memory_usage", 
                "cache_hit_rate",
                "query_success_rate",
                "vector_store_performance"
            ],
            "alerts": {
                "email_enabled": False,
                "log_enabled": True,
                "threshold_checks": True
            }
        },
        "performance_targets": {
            "average_response_time_ms": 2000,
            "cache_hit_rate_percent": 80,
            "success_rate_percent": 95,
            "memory_usage_limit_mb": 1000
        }
    }
    
    config_path = "data/performance_monitoring_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Performance monitoring config saved: {config_path}")
    return config

def setup_logging_optimization():
    """Configure optimized logging"""
    print_header("Logging Optimizations")
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/discord_bot.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "DEBUG",
                "formatter": "detailed"
            },
            "performance": {
                "class": "logging.FileHandler",
                "filename": "logs/performance.log",
                "level": "INFO",
                "formatter": "standard"
            }
        },
        "loggers": {
            "agentic": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "performance": {
                "level": "INFO", 
                "handlers": ["performance"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    config_path = "data/logging_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Logging config saved: {config_path}")
    return config

def create_unified_performance_config():
    """Create a unified configuration file for all optimizations"""
    print_header("Unified Performance Configuration")
    
    # Get all optimized configs
    vector_config = optimize_vector_store_config()
    memory_config = optimize_memory_settings()
    discord_config = optimize_discord_settings()
    analytics_config = optimize_analytics_settings()
    monitoring_config = create_performance_monitoring_config()
    
    unified_config = {
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
        "performance_optimizations": {
            "vector_store": vector_config,
            "memory": memory_config,
            "discord": discord_config,
            "analytics": analytics_config,
            "monitoring": monitoring_config
        },
        "feature_flags": {
            "enable_caching": True,
            "enable_compression": True,
            "enable_batch_processing": True,
            "enable_performance_monitoring": True,
            "enable_rate_limiting": True
        }
    }
    
    config_path = "data/unified_performance_config.json"
    with open(config_path, "w") as f:
        json.dump(unified_config, f, indent=2)
    
    print(f"✅ Unified performance config saved: {config_path}")
    return unified_config

def create_startup_script():
    """Create an optimized startup script"""
    print_header("Optimized Startup Script")
    
    startup_script = '''#!/usr/bin/env python3
"""
Optimized Discord Bot Startup Script
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Load optimized configurations
def load_performance_config():
    config_path = Path("data/unified_performance_config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def setup_optimized_logging():
    logging_config_path = Path("data/logging_config.json")
    if logging_config_path.exists():
        import logging.config
        with open(logging_config_path) as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

async def main():
    """Main optimized startup function"""
    print("🚀 Starting Discord Bot with Performance Optimizations")
    
    # Setup optimized logging
    setup_optimized_logging()
    logger = logging.getLogger("agentic.startup")
    
    # Load performance configurations
    perf_config = load_performance_config()
    logger.info("✅ Performance configurations loaded")
    
    # Import and start main bot
    from main import main as bot_main
    await bot_main()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    script_path = "start_optimized.py"
    with open(script_path, "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Optimized startup script created: {script_path}")

def main():
    """Main function to apply all optimizations"""
    print("⚡ Discord Bot Performance Optimization Suite")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Apply all optimizations
        setup_logging_optimization()
        unified_config = create_unified_performance_config()
        create_startup_script()
        
        print(f"\n{'=' * 60}")
        print("🎉 Performance Optimizations Applied!")
        print("✅ All configuration files created")
        print("✅ Optimized startup script ready")
        
        print("\n📊 Optimization Summary:")
        print("   • Vector store: Increased batch sizes, connection pooling")
        print("   • Memory: Compression, caching, connection pooling")
        print("   • Discord: Timeout protection, rate limiting")
        print("   • Analytics: Async recording, batch processing")
        print("   • Monitoring: Performance tracking, alerting")
        print("   • Logging: Rotation, structured output")
        
        print("\n🚀 Ready to start with optimizations:")
        print("   python start_optimized.py")
        print("   OR")
        print("   python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying optimizations: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
