"""
Conversation Memory

Manages conversation history and context for users across sessions.
"""

import asyncio
import json
import sqlite3
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
        
        self._init_database()
        logger.info("Conversation memory initialized")
    
    def _init_database(self):
        """Initialize the conversation memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                context TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes separately for SQLite
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_context (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                frequent_queries TEXT,
                last_active DATETIME,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def add_interaction(
        self, 
        user_id: str, 
        query: str, 
        response: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO conversations 
                (user_id, query, response, timestamp, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                query,
                response, 
                datetime.utcnow().isoformat(),
                json.dumps(context or {}),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            
            # Update user context
            await self._update_user_context(user_id, query)
            
            # Cleanup old conversations if needed
            await self._cleanup_old_conversations(user_id)
            
        except Exception as e:
            logger.error(f"Error adding interaction: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    async def get_history(
        self, 
        user_id: str, 
        limit: Optional[int] = None,
        hours_back: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return
            hours_back: Only return interactions from this many hours back
            
        Returns:
            List of conversation interactions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "response": row[1],
                    "timestamp": row[2],
                    "context": json.loads(row[3]) if row[3] else {},
                    "metadata": json.loads(row[4]) if row[4] else {}
                })
            
            return list(reversed(history))  # Return in chronological order
            
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
        finally:
            conn.close()
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get stored context for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT preferences, frequent_queries, last_active, metadata
                FROM user_context
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            return {
                "preferences": json.loads(row[0]) if row[0] else {},
                "frequent_queries": json.loads(row[1]) if row[1] else [],
                "last_active": row[2],
                "metadata": json.loads(row[3]) if row[3] else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting user context: {str(e)}")
            return {}
        finally:
            conn.close()
    
    async def _update_user_context(self, user_id: str, query: str):
        """Update user context with new interaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            cursor.execute("""
                INSERT OR REPLACE INTO user_context
                (user_id, preferences, frequent_queries, last_active, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                json.dumps(context.get("preferences", {})),
                json.dumps(frequent_queries),
                datetime.utcnow().isoformat(),
                json.dumps(context.get("metadata", {}))
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating user context: {str(e)}")
        finally:
            conn.close()
    
    async def _cleanup_old_conversations(self, user_id: str):
        """Remove old conversations beyond the limit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Keep only the most recent conversations
            cursor.execute("""
                DELETE FROM conversations
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, (user_id, user_id, self.max_history_length))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {str(e)}")
        finally:
            conn.close()
    
    async def clear_history(self, user_id: str):
        """Clear all conversation history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM user_context WHERE user_id = ?", (user_id,))
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
        finally:
            conn.close()
    
    async def get_conversation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    MIN(timestamp) as first_conversation,
                    MAX(timestamp) as last_conversation
                FROM conversations
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            
            return {
                "total_conversations": row[0] if row else 0,
                "first_conversation": row[1] if row else None,
                "last_conversation": row[2] if row else None
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation stats: {str(e)}")
            return {}
        finally:
            conn.close()
