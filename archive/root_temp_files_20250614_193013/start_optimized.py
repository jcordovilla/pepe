#!/usr/bin/env python3
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
