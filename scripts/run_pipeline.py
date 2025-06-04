#!/usr/bin/env python3
"""
Pipeline Runner - Restored Standalone Functionality
Bridges the gap between real-time Discord bot processing and manual data operations

DUAL APPROACH:
1. Real-time processing via Discord bot (main.py)
2. Standalone pipeline for manual operations (THIS SCRIPT)

The standalone pipeline is useful for:
- Initial data migration and setup
- Batch processing of historical data
- Manual data maintenance and cleanup
- Testing and development workflows
- Offline data processing

Both approaches can coexist and complement each other.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run the standalone pipeline with user guidance"""
    print("🔄 Discord Bot Data Processing Pipeline")
    print("=====================================")
    print("")
    print("📋 Available approaches:")
    print("")
    print("1️⃣  REAL-TIME PROCESSING (Recommended for ongoing operations):")
    print("   python main.py")
    print("   • Automatic message processing via Discord events")
    print("   • Real-time vector store updates")
    print("   • Conversation memory management")
    print("   • Live analytics tracking")
    print("")
    print("2️⃣  STANDALONE PIPELINE (For manual operations):")
    print("   python scripts/run_standalone_pipeline.py")
    print("   • Manual Discord message fetching")
    print("   • Batch content processing and classification")
    print("   • Embedding generation and vector store updates")
    print("   • Resource identification and analysis")
    print("   • Data synchronization and validation")
    print("")
    print("🎯 Pipeline modes:")
    print("   --mode process_existing    # Process existing JSON files (default)")
    print("   --mode fetch_and_process   # Fetch from Discord + process")
    print("   --mode fetch_only          # Only fetch messages")
    print("")
    print("📖 Examples:")
    print("   # Process existing data files")
    print("   python scripts/run_standalone_pipeline.py")
    print("")
    print("   # Fetch and process from specific guild")
    print("   python scripts/run_standalone_pipeline.py --mode fetch_and_process --guild-id 1353058864810950737")
    print("")
    print("   # Process specific channels")
    print("   python scripts/run_standalone_pipeline.py --mode fetch_and_process --guild-id 1353058864810950737 --channel-ids 1353449106537713756 1353448986408779877")
    print("")
    
    # Check if user wants to run the standalone pipeline
    if len(sys.argv) > 1:
        # User provided arguments, run standalone pipeline
        standalone_script = Path(__file__).parent / "run_standalone_pipeline.py"
        
        if standalone_script.exists():
            print("🚀 Running standalone pipeline with provided arguments...")
            cmd = [sys.executable, str(standalone_script)] + sys.argv[1:]
            return subprocess.call(cmd)
        else:
            print("❌ Standalone pipeline script not found!")
            return 1
    else:
        # No arguments provided, show help
        print("💡 To run the standalone pipeline, add arguments:")
        print("   python scripts/run_pipeline.py --mode process_existing")
        print("")
        print("💡 Or run the Discord bot for real-time processing:")
        print("   python main.py")
        print("")
        return 0

if __name__ == "__main__":
    sys.exit(main())
