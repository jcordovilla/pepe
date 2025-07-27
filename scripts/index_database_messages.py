#!/usr/bin/env python3
"""
Database Message Validator

Simple script to validate the SQLite database and show statistics.
No longer needed for ChromaDB indexing since we use MCP server.
"""

import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.config.modernized_config import get_modernized_config

class DatabaseValidator:
    def __init__(self):
        self.db_path = project_root / 'data' / 'discord_messages.db'
        self.config = get_modernized_config()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_database(self):
        """Validate the SQLite database and show statistics"""
        print("🔍 Validating Discord messages database...", flush=True)
        print(f"📊 Database: {self.db_path}", flush=True)
        
        if not self.db_path.exists():
            print("❌ Database file not found!")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
                if not cursor.fetchone():
                    print("❌ 'messages' table not found!")
                    return False
                
                # Get basic statistics
                cursor.execute("SELECT COUNT(*) FROM messages")
                total_messages = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT channel_id) FROM messages")
                unique_channels = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT author_id) FROM messages")
                unique_users = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
                min_date, max_date = cursor.fetchone()
                
                # Get top channels
                cursor.execute("""
                    SELECT channel_name, COUNT(*) as message_count 
                    FROM messages 
                    GROUP BY channel_id, channel_name 
                    ORDER BY message_count DESC 
                    LIMIT 10
                """)
                top_channels = cursor.fetchall()
                
                # Get top users
                cursor.execute("""
                    SELECT author_username, COUNT(*) as message_count 
                    FROM messages 
                    GROUP BY author_id, author_username 
                    ORDER BY message_count DESC 
                    LIMIT 10
                """)
                top_users = cursor.fetchall()
                
                # Display results
                print(f"✅ Database validation successful!")
                print(f"📝 Total messages: {total_messages:,}")
                print(f"📺 Unique channels: {unique_channels}")
                print(f"👥 Unique users: {unique_users}")
                print(f"📅 Date range: {min_date} to {max_date}")
                
                print(f"\n🏆 Top 10 Channels:")
                for i, (channel_name, count) in enumerate(top_channels, 1):
                    print(f"  {i:2d}. {channel_name}: {count:,} messages")
                
                print(f"\n👑 Top 10 Users:")
                for i, (username, count) in enumerate(top_users, 1):
                    print(f"  {i:2d}. {username}: {count:,} messages")
                
                return True
                
        except Exception as e:
            print(f"❌ Database validation failed: {e}")
            return False
    
    def check_indices(self):
        """Check if database indices are properly created"""
        print(f"\n🔍 Checking database indices...", flush=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for important indices
                indices = [
                    "idx_messages_content_search",
                    "idx_messages_timestamp_range", 
                    "idx_messages_author_activity",
                    "idx_messages_channel_activity"
                ]
                
                for index_name in indices:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?", (index_name,))
                    if cursor.fetchone():
                        print(f"✅ {index_name}")
                    else:
                        print(f"⚠️  {index_name} (missing)")
                
                return True
                
        except Exception as e:
            print(f"❌ Index check failed: {e}")
            return False

async def main():
    """Main validation function"""
    validator = DatabaseValidator()
    
    # Validate database
    if not validator.validate_database():
        return
    
    # Check indices
    validator.check_indices()
    
    print(f"\n🎉 Database validation complete!")
    print(f"💡 The database is ready for use with the MCP server.")

if __name__ == "__main__":
    asyncio.run(main()) 