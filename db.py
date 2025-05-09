# db.py

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, create_engine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Engine points to a local SQLite file
engine = create_engine("sqlite:///discord_messages.db", echo=False)

# 2. Base class for models
Base = declarative_base()

# 3. Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Message(Base):
    __tablename__ = "messages"
    # Primary key auto-increments
    id = Column(Integer, primary_key=True, index=True)
    guild_id = Column(Integer, index=True, nullable=False)
    channel_id = Column(Integer, index=True, nullable=False)
    message_id = Column(Integer, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    author = Column(JSON, nullable=False)       # stores dict with id, username, discriminator
    mention_ids = Column(JSON, nullable=False)  # list of ints
    reactions = Column(JSON, nullable=False)    # list of {emoji, count}
    jump_url = Column(String, nullable=True)
    channel_name = Column(String, index=True, nullable=True)  # <<< add this

# 4. Create tables if they don't exist
Base.metadata.create_all(engine)
