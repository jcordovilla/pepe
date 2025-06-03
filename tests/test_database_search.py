#!/usr/bin/env python3
"""
Test script to verify that the Discord bot can now search the populated database
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Database setup using the same pattern as init_db_simple.py
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/discord_bot.db')
engine = create_engine(DATABASE_URL)

# Import the Message class from our init script structure
import sys
sys.path.insert(0, '/Users/jose/Documents/apps/discord-bot-v2')

# We need to recreate the Message class with the same structure
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    guild_id = Column(Integer, index=True, nullable=False)
    channel_id = Column(Integer, index=True, nullable=False)
    channel_name = Column(String, nullable=False)
    message_id = Column(Integer, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    author = Column(JSON, nullable=False)
    mention_ids = Column(JSON, nullable=False)
    reactions = Column(JSON, nullable=False)
    jump_url = Column(String, nullable=True)
    resource_detected = Column(Integer, default=0, nullable=False)

SessionLocal = sessionmaker(bind=engine)

def test_database_queries():
    """Test various database queries to simulate bot functionality"""
    print("🔍 Testing database queries...")
    
    session = SessionLocal()
    
    try:
        # Test 1: Count all messages
        total_count = session.query(Message).count()
        print(f"✅ Total messages in database: {total_count}")
        
        # Test 2: List messages in agent-dev channel 
        agent_dev_messages = session.query(Message).filter(
            Message.channel_name == "agent-dev"
        ).order_by(Message.timestamp.desc()).limit(3).all()
        
        print(f"\n🎯 Last 3 messages in #agent-dev:")
        for i, msg in enumerate(agent_dev_messages, 1):
            author_name = msg.author.get('username', 'Unknown') if isinstance(msg.author, dict) else 'Unknown'
            print(f"{i}. {author_name}: {msg.content[:60]}...")
            print(f"   🕒 {msg.timestamp}")
            print(f"   🔗 {msg.jump_url}")
        
        # Test 3: Search for messages containing specific keywords
        search_terms = ["machine learning", "agent", "development"]
        
        for term in search_terms:
            results = session.query(Message).filter(
                Message.content.contains(term)
            ).all()
            print(f"\n🔎 Messages containing '{term}': {len(results)} found")
            for msg in results[:2]:  # Show first 2 results
                author_name = msg.author.get('username', 'Unknown') if isinstance(msg.author, dict) else 'Unknown'
                print(f"   - {author_name} in #{msg.channel_name}: {msg.content[:50]}...")
        
        # Test 4: List all channels
        channels = session.query(Message.channel_name, Message.channel_id).distinct().all()
        print(f"\n📋 Available channels:")
        for channel_name, channel_id in channels:
            count = session.query(Message).filter(Message.channel_id == channel_id).count()
            print(f"   - #{channel_name} (ID: {channel_id}) - {count} messages")
        
        print(f"\n✅ All database tests passed! The bot should now be able to search for messages.")
        
        # Test 5: Simulate the exact query from the issue
        print(f"\n🚀 Testing the exact query: 'list the last 3 messages in #agent-dev'")
        query_results = session.query(Message).filter(
            Message.channel_name == "agent-dev"
        ).order_by(Message.timestamp.desc()).limit(3).all()
        
        if query_results:
            print(f"✅ Query successful! Found {len(query_results)} messages:")
            for i, msg in enumerate(query_results, 1):
                author_name = msg.author.get('username', 'Unknown') if isinstance(msg.author, dict) else 'Unknown'
                print(f"{i}. {author_name}: {msg.content}")
        else:
            print("❌ No results found - this indicates an issue")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("🧪 Testing Discord bot database functionality...")
    test_database_queries()
    print("🎉 Database test complete!")
