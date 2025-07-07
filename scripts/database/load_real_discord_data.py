#!/usr/bin/env python3
"""
Load Real Discord Message Data for Better Evaluation

This script loads real Discord message data from JSON files into the database
to replace synthetic test data with actual Discord conversations for more
accurate AI classification evaluation.
"""

import json
import sqlite3
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_discord_json_files(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all Discord message JSON files and extract messages."""
    all_messages = []
    json_files = list(data_dir.glob("*_messages.json"))
    
    logger.info(f"Found {len(json_files)} Discord message files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract messages from the JSON structure
            messages = data.get('messages', [])
            
            # Filter messages with actual content (not just join messages)
            content_messages = [
                msg for msg in messages 
                if msg.get('content', '').strip() and len(msg.get('content', '').strip()) > 10
            ]
            
            logger.info(f"Loaded {len(content_messages)} content messages from {json_file.name}")
            all_messages.extend(content_messages)
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Total loaded messages with content: {len(all_messages)}")
    return all_messages

def clear_existing_data(db_path: Path):
    """Clear existing synthetic test data from messages table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current message count
        cursor.execute("SELECT COUNT(*) FROM messages")
        before_count = cursor.fetchone()[0]
        logger.info(f"Messages in database before clearing: {before_count}")
        
        # Clear existing messages
        cursor.execute("DELETE FROM messages")
        conn.commit()
        
        logger.info("Cleared existing message data")
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise
    finally:
        conn.close()

def insert_discord_messages(db_path: Path, messages: List[Dict[str, Any]]):
    """Insert real Discord messages into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        inserted_count = 0
        
        for msg in messages:
            try:
                # Extract message data
                message_id = msg.get('message_id', '')
                channel_id = msg.get('channel_id', '')
                channel_name = msg.get('channel_name', '')
                guild_id = msg.get('guild_id', '')
                content = msg.get('content', '').strip()
                timestamp = msg.get('timestamp', '')
                jump_url = msg.get('jump_url', '')
                
                # Extract author info as JSON string (matching schema)
                author = msg.get('author', {})
                
                # Extract mentions and reactions as JSON
                mentions = msg.get('mentions', [])
                mention_ids = [mention.get('id', '') for mention in mentions if mention.get('id')]
                reactions = msg.get('reactions', [])
                
                # Skip if essential data is missing
                if not content or not message_id:
                    continue
                
                # Insert into database matching actual schema
                cursor.execute("""
                    INSERT INTO messages (
                        guild_id, channel_id, channel_name, message_id, content, 
                        timestamp, author, mention_ids, reactions, jump_url, resource_detected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    guild_id, channel_id, channel_name, message_id, content,
                    timestamp, json.dumps(author), json.dumps(mention_ids), 
                    json.dumps(reactions), jump_url, 0  # resource_detected = 0 initially
                ))
                
                inserted_count += 1
                
                if inserted_count % 100 == 0:
                    logger.info(f"Inserted {inserted_count} messages...")
                    
            except Exception as e:
                logger.warning(f"Error inserting message {msg.get('message_id', 'unknown')}: {e}")
                continue
        
        conn.commit()
        logger.info(f"Successfully inserted {inserted_count} real Discord messages")
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_count = cursor.fetchone()[0]
        logger.info(f"Total messages in database: {total_count}")
        
        return inserted_count
        
    except Exception as e:
        logger.error(f"Error inserting messages: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def verify_data_quality(db_path: Path):
    """Verify the quality and variety of loaded data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Messages with URLs
        cursor.execute("SELECT COUNT(*) FROM messages WHERE content LIKE '%http%'")
        url_messages = cursor.fetchone()[0]
        
        # Messages with code patterns
        cursor.execute("SELECT COUNT(*) FROM messages WHERE content LIKE '%```%' OR content LIKE '%`%'")
        code_messages = cursor.fetchone()[0]
        
        # Messages with attachments (check content for attachment indicators)
        cursor.execute("SELECT COUNT(*) FROM messages WHERE content LIKE '%attachment%' OR content LIKE '%image%' OR content LIKE '%file%'")
        attachment_messages = cursor.fetchone()[0]
        
        # Long messages (likely detailed content)
        cursor.execute("SELECT COUNT(*) FROM messages WHERE LENGTH(content) > 200")
        long_messages = cursor.fetchone()[0]
        
        # Sample some content types
        cursor.execute("""
            SELECT content, LENGTH(content) as len 
            FROM messages 
            WHERE content LIKE '%http%' 
            LIMIT 3
        """)
        url_samples = cursor.fetchall()
        
        cursor.execute("""
            SELECT content, LENGTH(content) as len 
            FROM messages 
            WHERE LENGTH(content) > 500 
            LIMIT 3
        """)
        long_samples = cursor.fetchall()
        
        # Print verification report
        logger.info("="*60)
        logger.info("REAL DISCORD DATA VERIFICATION REPORT")
        logger.info("="*60)
        logger.info(f"Total messages loaded: {total_messages}")
        logger.info(f"Messages with URLs: {url_messages} ({url_messages/total_messages*100:.1f}%)")
        logger.info(f"Messages with code: {code_messages} ({code_messages/total_messages*100:.1f}%)")
        logger.info(f"Messages with attachments: {attachment_messages} ({attachment_messages/total_messages*100:.1f}%)")
        logger.info(f"Long messages (>200 chars): {long_messages} ({long_messages/total_messages*100:.1f}%)")
        logger.info("")
        
        logger.info("URL Message Samples:")
        for i, (content, length) in enumerate(url_samples, 1):
            logger.info(f"  {i}. [{length} chars] {content[:100]}...")
        
        logger.info("")
        logger.info("Long Message Samples:")
        for i, (content, length) in enumerate(long_samples, 1):
            logger.info(f"  {i}. [{length} chars] {content[:100]}...")
        
        logger.info("="*60)
        
        return {
            'total_messages': total_messages,
            'url_messages': url_messages,
            'code_messages': code_messages,
            'attachment_messages': attachment_messages,
            'long_messages': long_messages
        }
        
    except Exception as e:
        logger.error(f"Error verifying data: {e}")
        raise
    finally:
        conn.close()

def main():
    """Main function to load real Discord data."""
    try:
        # Paths
        data_dir = project_root / "data" / "fetched_messages"
        db_path = project_root / "data" / "discord_bot.db"
        
        # Verify paths exist
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return False
            
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            return False
        
        logger.info("🚀 Loading Real Discord Message Data")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🗄️  Database: {db_path}")
        logger.info("")
        
        # Step 1: Load Discord messages from JSON files
        logger.info("📥 Step 1: Loading Discord message files...")
        messages = load_discord_json_files(data_dir)
        
        if not messages:
            logger.error("No messages loaded from JSON files")
            return False
        
        # Step 2: Clear existing synthetic data
        logger.info("🧹 Step 2: Clearing existing synthetic data...")
        clear_existing_data(db_path)
        
        # Step 3: Insert real Discord messages
        logger.info("💾 Step 3: Inserting real Discord messages...")
        inserted_count = insert_discord_messages(db_path, messages)
        
        if inserted_count == 0:
            logger.error("No messages were inserted")
            return False
        
        # Step 4: Verify data quality
        logger.info("✅ Step 4: Verifying data quality...")
        stats = verify_data_quality(db_path)
        
        logger.info("")
        logger.info("🎉 Real Discord data loading completed successfully!")
        logger.info(f"📊 Loaded {inserted_count} real Discord messages")
        logger.info("🔬 Ready for improved evaluation testing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
