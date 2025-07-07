"""
Conversation Memory

Manages conversation history and context for users across sessions.
"""

import asyncio
import json

try:
    import aiosqlite
except ImportError:  # Fallback for environments without aiosqlite
    import sqlite3
    import types

    class _AsyncCursor:
        def __init__(self, cur):
            self._cur = cur

        async def execute(self, *args, **kwargs):
            return self._cur.execute(*args, **kwargs)

        async def fetchall(self):
            return self._cur.fetchall()

        async def fetchone(self):
            return self._cur.fetchone()

    class _AsyncConnection:
        def __init__(self, conn):
            self._conn = conn

        async def cursor(self):
            return _AsyncCursor(self._conn.cursor())

        async def commit(self):
            self._conn.commit()

        async def rollback(self):
            self._conn.rollback()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self._conn.close()

    def connect(path):
        return _AsyncConnection(sqlite3.connect(path))

    aiosqlite = types.SimpleNamespace(connect=connect)
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and user context.

    Stores conversations in SQLite and provides efficient retrieval
    with context management and summarization capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("db_path", "data/conversation_memory.db")
        self.max_history_length = config.get("max_history_length", 50)
        self.context_window_hours = config.get("context_window_hours", 24)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._init_task = loop.create_task(self._init_database())
        else:
            loop.run_until_complete(self._init_database())
            self._init_task = None
        logger.info("Conversation memory initialized")

    async def _init_database(self):
        """Initialize the conversation memory database"""
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            await cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                context TEXT,
                metadata TEXT
            )
        """
            )

            await cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """
            )

            # Create indexes separately for SQLite
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")

            await cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS user_context (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                frequent_queries TEXT,
                last_active DATETIME,
                metadata TEXT
            )
        """
            )

            await conn.commit()

    async def _store_summary(self, user_id: str, summary: str):
        """Store a conversation summary in the database"""
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                INSERT INTO conversation_summaries (user_id, summary, timestamp)
                VALUES (?, ?, ?)
                """,
                (user_id, summary, datetime.utcnow().isoformat()),
            )
            await conn.commit()

    async def _summarize_history(self, interactions: List[Dict[str, Any]]) -> str:
        """Summarize a list of interactions using a simple heuristic"""
        if not interactions:
            return ""

        text = " ".join(f"{item.get('query', '')} {item.get('response', '')}" for item in interactions)
        words = text.split()
        summary = " ".join(words[:100])
        if len(words) > 100:
            summary += "..."
        return summary

    async def add_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a new conversation interaction.

        Args:
            user_id: User identifier
            query: User's query
            response: System response
            context: Additional context
            metadata: Interaction metadata
        """
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                await cursor.execute(
                    """
                INSERT INTO conversations
                (user_id, query, response, timestamp, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                    (
                        user_id,
                        query,
                        response,
                        datetime.utcnow().isoformat(),
                        json.dumps(context or {}),
                        json.dumps(metadata or {}),
                    ),
                )

                await conn.commit()

                # Update user context
                await self._update_user_context(user_id, query)

                # Cleanup old conversations if needed
                await self._cleanup_old_conversations(user_id)

            except Exception as e:
                logger.error(f"Error adding interaction: {str(e)}")
                await conn.rollback()

    async def get_history(
        self, user_id: str, limit: Optional[int] = None, hours_back: Optional[int] = None, summarize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return
            hours_back: Only return interactions from this many hours back
            summarize: Return a summary if True

        Returns:
            List of conversation interactions or a summary with recent history
        """
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                query = """
                    SELECT query, response, timestamp, context, metadata
                    FROM conversations
                    WHERE user_id = ?
                """
                params: List[Any] = [user_id]

                if hours_back:
                    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
                    query += " AND timestamp > ?"
                    params.append(cutoff.isoformat())

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                await cursor.execute(query, params)
                rows = await cursor.fetchall()

                history = []
                for row in rows:
                    history.append(
                        {
                            "query": row[0],
                            "response": row[1],
                            "timestamp": row[2],
                            "context": json.loads(row[3]) if row[3] else {},
                            "metadata": json.loads(row[4]) if row[4] else {},
                        }
                    )

                history = list(reversed(history))

                if summarize or len(history) > self.max_history_length:
                    to_summarize = (
                        history[: -self.max_history_length] if len(history) > self.max_history_length else history
                    )
                    summary = await self._summarize_history(to_summarize)
                    await self._store_summary(user_id, summary)
                    trimmed = history[-self.max_history_length :]
                    return [{"summary": summary}] + trimmed

                return history  # Return in chronological order

            except Exception as e:
                logger.error(f"Error getting history: {str(e)}")
                return []

    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get stored context for a user.

        Args:
            user_id: User identifier

        Returns:
            User context dictionary
        """
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                await cursor.execute(
                    """
                SELECT preferences, frequent_queries, last_active, metadata
                FROM user_context
                WHERE user_id = ?
            """,
                    (user_id,),
                )

                row = await cursor.fetchone()
                if not row:
                    return {}

                return {
                    "preferences": json.loads(row[0]) if row[0] else {},
                    "frequent_queries": json.loads(row[1]) if row[1] else [],
                    "last_active": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {},
                }

            except Exception as e:
                logger.error(f"Error getting user context: {str(e)}")
                return {}

    async def _update_user_context(self, user_id: str, query: str):
        """Update user context with new interaction"""
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                # Get current context
                context = await self.get_user_context(user_id)

                # Update frequent queries
                frequent_queries = context.get("frequent_queries", [])
                # Simple frequency tracking - can be enhanced
                query_lower = query.lower()
                if query_lower not in frequent_queries:
                    frequent_queries.append(query_lower)
                    # Keep only last 20 queries
                    frequent_queries = frequent_queries[-20:]

                # Upsert user context
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO user_context
                    (user_id, preferences, frequent_queries, last_active, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        json.dumps(context.get("preferences", {})),
                        json.dumps(frequent_queries),
                        datetime.utcnow().isoformat(),
                        json.dumps(context.get("metadata", {})),
                    ),
                )

                await conn.commit()

            except Exception as e:
                logger.error(f"Error updating user context: {str(e)}")

    async def _cleanup_old_conversations(self, user_id: str):
        """Remove old conversations beyond the limit"""
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                # Keep only the most recent conversations
                await cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE user_id = ? AND id NOT IN (
                        SELECT id FROM conversations
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                """,
                    (user_id, user_id, self.max_history_length),
                )

                await conn.commit()

            except Exception as e:
                logger.error(f"Error cleaning up conversations: {str(e)}")

    async def clear_history(self, user_id: str):
        """Clear all conversation history for a user"""
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                await cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                await cursor.execute("DELETE FROM user_context WHERE user_id = ?", (user_id,))
                await conn.commit()

            except Exception as e:
                logger.error(f"Error clearing history: {str(e)}")

    async def get_conversation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        if getattr(self, "_init_task", None):
            await self._init_task
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.cursor()

            try:
                await cursor.execute(
                    """
                SELECT
                    COUNT(*) as total_conversations,
                    MIN(timestamp) as first_conversation,
                    MAX(timestamp) as last_conversation
                FROM conversations
                WHERE user_id = ?
            """,
                    (user_id,),
                )

                row = await cursor.fetchone()

                return {
                    "total_conversations": row[0] if row else 0,
                    "first_conversation": row[1] if row else None,
                    "last_conversation": row[2] if row else None,
                }

            except Exception as e:
                logger.error(f"Error getting conversation stats: {str(e)}")
                return {}
