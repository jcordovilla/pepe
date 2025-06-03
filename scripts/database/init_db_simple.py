#!/usr/bin/env python3
"""
Simple database initialization script that creates tables and adds sample data
"""

import os
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/discord_bot.db')
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

# Message model definition
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

# Create session
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def create_sample_data():
    """Create tables and sample Discord messages for testing"""
    print("🔄 Creating database tables...")
    
    # Create all tables
    Base.metadata.create_all(engine)
    print("✅ Tables created successfully")
    
    session = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = session.query(Message).count()
        if existing_count > 0:
            print(f"Database already has {existing_count} messages. Skipping initialization.")
            return
        
        print("📝 Adding sample messages...")
        
        # Sample channels
        channels = [
            {"id": 1001, "name": "agent-dev"},
            {"id": 1002, "name": "general-chat"},
            {"id": 1003, "name": "announcements"},
            {"id": 1004, "name": "tech-discussions"},
            {"id": 1005, "name": "q-and-a"}
        ]
        
        # Sample messages for agent-dev channel (most important for testing)
        sample_messages = [
            {
                "content": "Welcome to the agent development channel! This is where we discuss AI agents and automation.",
                "author": {"id": "12345", "username": "bot_admin", "display_name": "Bot Admin"},
                "channel": channels[0]
            },
            {
                "content": "I have been working on a new machine learning model for text classification. The results look promising!",
                "author": {"id": "12346", "username": "ml_engineer", "display_name": "ML Engineer"},
                "channel": channels[0]
            },
            {
                "content": "The last 3 messages in this channel have been about development topics. Looking forward to more discussions!",
                "author": {"id": "12349", "username": "community_mod", "display_name": "Community Mod"},
                "channel": channels[0]
            },
            {
                "content": "I found this amazing AI research paper on transformer architectures. Worth reading!",
                "author": {"id": "12352", "username": "ai_researcher", "display_name": "AI Researcher"},
                "channel": channels[0]
            },
            {
                "content": "Can someone help me understand how async programming works in Python?",
                "author": {"id": "12347", "username": "python_learner", "display_name": "Python Learner"},
                "channel": channels[1]
            },
            {
                "content": "Important announcement: We're updating our Discord bot with new features!",
                "author": {"id": "12350", "username": "admin", "display_name": "Admin"},
                "channel": channels[2]
            }
        ]
        
        # Insert sample messages
        base_time = datetime.now() - timedelta(hours=6)
        
        for i, msg_data in enumerate(sample_messages):
            # Create message timestamp (spread over last 6 hours)
            timestamp = base_time + timedelta(hours=i)
            
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
        print("\n📊 Database Summary:")
        for channel in channels:
            count = session.query(Message).filter(Message.channel_id == channel["id"]).count()
            if count > 0:
                print(f"- #{channel['name']} (ID: {channel['id']}) - {count} messages")
                
        # Verify total count
        total_messages = session.query(Message).count()
        print(f"\n🎯 Total messages in database: {total_messages}")
        
        # Show some sample data from agent-dev channel for verification
        agent_dev_messages = session.query(Message).filter(Message.channel_name == "agent-dev").limit(3).all()
        print(f"\n🔍 Sample messages from #agent-dev:")
        for msg in agent_dev_messages:
            print(f"- {msg.author.get('username', 'Unknown')}: {msg.content[:50]}...")
            
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("🚀 Initializing Discord bot database with sample data...")
    create_sample_data()
    print("🎉 Database initialization complete!")
