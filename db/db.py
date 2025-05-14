# db.py

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
import logging
from typing import Optional
from contextlib import contextmanager
import time
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying database operations.
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise last_exception
            return None
        return wrapper
    return decorator

# 1. Engine points to a local SQLite file with connection pooling
engine = create_engine(
    "sqlite:///data/discord_messages.db",
    echo=False,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800  # Recycle connections after 30 minutes
)

# 2. Base class for models
Base = declarative_base()

# 3. Session factory with retry mechanism
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

class Message(Base):
    __tablename__ = "messages"
    # Primary key auto-increments
    id = Column(Integer, primary_key=True, index=True)
    guild_id = Column(Integer, index=True, nullable=False)
    channel_id = Column(Integer, index=True, nullable=False)
    channel_name = Column(String, nullable=False)
    message_id = Column(Integer, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    author = Column(JSON, nullable=False)       # stores dict with id, username, discriminator
    mention_ids = Column(JSON, nullable=False)  # list of ints
    reactions = Column(JSON, nullable=False)    # list of {emoji, count}
    jump_url = Column(String, nullable=True)

class Resource(Base):
    __tablename__ = "resources"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, index=True, nullable=False)
    guild_id   = Column(Integer, index=True, nullable=False)
    channel_id = Column(Integer, index=True, nullable=False)
    url        = Column(String, nullable=False)
    type       = Column(String, nullable=True)   # e.g. "PDF", "article"
    tag        = Column(String, nullable=True)   # e.g. "Paper", "Tutorial"
    author     = Column(JSON,   nullable=True)   # reuse same JSON structure as Message.author
    author_display = Column(String, nullable=True)  # display/global/nick name
    channel_name = Column(String, nullable=True)    # channel name
    timestamp  = Column(DateTime, nullable=False)
    context_snippet = Column(Text, nullable=True)
    meta   = Column(JSON,   nullable=True)   # any extra parsed fields (renamed from metadata)
    name = Column(String, nullable=True)  # new: resource name/title
    description = Column(Text, nullable=True)  # new: resource description
    jump_url = Column(String, nullable=True)  # new: jump_url for direct Discord link

@contextmanager
def get_db_session():
    """
    Context manager for database sessions with automatic cleanup.
    Usage:
        with get_db_session() as session:
            # Use session here
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

@with_retry(max_retries=3)
def execute_query(query_func):
    """
    Execute a database query with retry mechanism.
    Args:
        query_func: Function that takes a session and returns query results
    """
    with get_db_session() as session:
        return query_func(session)

# 4. Create tables if they don't exist
Base.metadata.create_all(engine)
