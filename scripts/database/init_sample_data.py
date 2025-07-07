#!/usr/bin/env python3
"""
Initialize the Discord bot database with sample data for testing
"""
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
from db import SessionLocal, Message  
from db.db import Base
import json

def create_sample_data():
    """Create sample Discord messages for testing"""
    
    # Load environment
    load_dotenv()
    
    # Create database tables
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/discord_bot.db')
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    
    session = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = session.query(Message).count()
        if existing_count > 0:
            print(f"Database already has {existing_count} messages. Skipping initialization.")
            return
        
        # Sample channels
        channels = [
            {"id": 1001, "name": "agent-dev"},
            {"id": 1002, "name": "general-chat"},
            {"id": 1003, "name": "announcements"},
            {"id": 1004, "name": "tech-discussions"},
            {"id": 1005, "name": "q-and-a"}
        ]
        
        # Sample messages
        sample_messages = [
            {
                "content": "Welcome to the agent development channel! This is where we discuss AI agents and automation.",
                "author": {"id": "12345", "username": "bot_admin", "display_name": "Bot Admin"},
                "channel": channels[0]
            },
            {
                "content": "I've been working on a new machine learning model for text classification. The results look promising!",
                "author": {"id": "12346", "username": "ml_engineer", "display_name": "ML Engineer"},
                "channel": channels[0]
            },
            {
                "content": "Can someone help me understand how async programming works in Python?",
                "author": {"id": "12347", "username": "python_learner", "display_name": "Python Learner"},
                "channel": channels[1]
            },
            {
                "content": "Here's a great tutorial on Docker containers: https://docker.com/tutorial",
                "author": {"id": "12348", "username": "devops_expert", "display_name": "DevOps Expert"},
                "channel": channels[3]
            },
            {
                "content": "The last 3 messages in this channel have been about development topics. Looking forward to more discussions!",
                "author": {"id": "12349", "username": "community_mod", "display_name": "Community Mod"},
                "channel": channels[0]
            },
            {
                "content": "Important announcement: We're updating our Discord bot with new features!",
                "author": {"id": "12350", "username": "admin", "display_name": "Admin"},
                "channel": channels[2]
            },
            {
                "content": "What are the best practices for API design? I'm working on a REST API for our project.",
                "author": {"id": "12351", "username": "backend_dev", "display_name": "Backend Dev"},
                "channel": channels[4]
            },
            {
                "content": "I found this amazing AI research paper on transformer architectures. Worth reading!",
                "author": {"id": "12352", "username": "ai_researcher", "display_name": "AI Researcher"},
                "channel": channels[0]
            }
        ]
        
        # Insert sample messages
        base_time = datetime.now() - timedelta(days=2)
        
        for i, msg_data in enumerate(sample_messages):
            # Create message timestamp (spread over last 2 days)
            timestamp = base_time + timedelta(hours=i * 3)
            
            message = Message(
                guild_id=999,  # Sample guild ID
                channel_id=msg_data["channel"]["id"],
                channel_name=msg_data["channel"]["name"],
                message_id=2000 + i,  # Sample message ID
                content=msg_data["content"],
                timestamp=timestamp,
                author=msg_data["author"],
                mention_ids=[],
                reactions=[],
                jump_url=f"https://discord.com/channels/999/{msg_data['channel']['id']}/{2000 + i}",
                resource_detected=0
            )
            
            session.add(message)
        
        session.commit()
        print(f"✅ Successfully created {len(sample_messages)} sample messages")
        
        # Print summary
        print("\nChannels created:")
        for channel in channels:
            count = session.query(Message).filter(Message.channel_id == channel["id"]).count()
            print(f"- #{channel['name']} (ID: {channel['id']}) - {count} messages")
            
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("🔄 Initializing database with sample data...")
    create_sample_data()
    print("🎉 Database initialization complete!")
