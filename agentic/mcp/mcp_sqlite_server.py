"""
MCP SQLite Server Implementation

Uses the mcp-sqlite library to provide standardized SQLite query capabilities
for the agentic Discord bot system.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPSQLiteServer:
    """
    MCP SQLite server implementation using the mcp-sqlite library.
    
    Provides standardized SQLite query capabilities with:
    - Natural language to SQL translation
    - Direct database queries
    - Schema introspection
    - Query optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.database_path = self.config.get("database_path", "data/discord_messages.db")
        self.enable_write = self.config.get("enable_write", False)
        self.metadata_path = self.config.get("metadata_path")
        self.verbose = self.config.get("verbose", False)
        
        # Process management
        self._process: Optional[subprocess.Popen] = None
        self._started = False
        
        logger.info(f"MCPSQLiteServer initialized with database: {self.database_path}")
    
    async def start(self):
        """Start the MCP SQLite server as a subprocess."""
        if self._started:
            logger.warning("MCP SQLite server already started")
            return
        
        try:
            # Build command
            cmd = [
                "python", "-m", "mcp_sqlite.server",
                self.database_path
            ]
            
            if self.enable_write:
                cmd.append("--write")
            
            if self.metadata_path:
                cmd.extend(["--metadata", self.metadata_path])
            
            if self.verbose:
                cmd.append("--verbose")
            
            logger.info(f"Starting MCP SQLite server: {' '.join(cmd)}")
            
            # Start the subprocess
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Wait a moment for the server to start
            await asyncio.sleep(1)
            
            if self._process.poll() is None:
                self._started = True
                logger.info("MCP SQLite server started successfully")
            else:
                raise RuntimeError("MCP SQLite server failed to start")
                
        except Exception as e:
            logger.error(f"Failed to start MCP SQLite server: {e}")
            if self._process:
                self._process.terminate()
            raise
    
    async def stop(self):
        """Stop the MCP SQLite server."""
        if not self._started or not self._process:
            return
        
        try:
            logger.info("Stopping MCP SQLite server...")
            self._process.terminate()
            
            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing MCP SQLite server")
                self._process.kill()
                self._process.wait()
            
            self._started = False
            self._process = None
            logger.info("MCP SQLite server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP SQLite server: {e}")
    
    async def query_messages(self, query: str) -> List[Dict[str, Any]]:
        """
        Query messages using natural language.
        
        Since mcp-sqlite runs as a separate process, we'll use a simplified
        approach that translates natural language to SQL and executes it.
        """
        try:
            # For now, use a simple approach - translate to SQL and execute
            sql_query = await self._translate_to_sql(query)
            return await self._execute_sql(sql_query)
        except Exception as e:
            logger.error(f"Error querying messages: {e}")
            return []
    
    async def search_messages(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search messages using text search."""
        try:
            # Build SQL query for text search
            sql_query = self._build_search_sql(query, filters, limit)
            return await self._execute_sql(sql_query)
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        try:
            sql_query = """
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            return await self._execute_sql(sql_query)
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            if not self._started or not self._process:
                return {"status": "stopped", "error": "Server not running"}
            
            if self._process.poll() is not None:
                return {"status": "dead", "error": "Process terminated"}
            
            # Simple health check query
            result = await self._execute_sql("SELECT 1 as health_check")
            if result:
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}
            else:
                return {"status": "unhealthy", "error": "Query failed"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _translate_to_sql(self, query: str) -> str:
        """Translate natural language query to SQL."""
        # Simple translation logic - in a real implementation, this would use an LLM
        query_lower = query.lower()
        
        if "recent" in query_lower or "latest" in query_lower:
            return "SELECT * FROM messages ORDER BY timestamp_unix DESC LIMIT 10"
        elif "count" in query_lower:
            return "SELECT COUNT(*) as count FROM messages"
        elif "show me" in query_lower and "messages" in query_lower:
            return "SELECT * FROM messages ORDER BY timestamp_unix DESC LIMIT 5"
        else:
            # Default query
            return "SELECT * FROM messages ORDER BY timestamp_unix DESC LIMIT 10"
    
    def _build_search_sql(self, query: str, filters: Optional[Dict] = None, limit: int = 10) -> str:
        """Build SQL query for text search."""
        sql = "SELECT * FROM messages WHERE 1=1"
        
        # Add text search
        if query:
            sql += f" AND content LIKE '%{query}%'"
        
        # Add filters
        if filters:
            if filters.get("author_bot") is False:
                sql += " AND author_bot = 0"
            if filters.get("channel_id"):
                sql += f" AND channel_id = {filters['channel_id']}"
        
        sql += f" ORDER BY timestamp_unix DESC LIMIT {limit}"
        return sql
    
    async def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query directly using aiosqlite."""
        import aiosqlite
        
        try:
            async with aiosqlite.connect(self.database_path) as db:
                async with db.execute(sql) as cursor:
                    columns = [description[0] for description in cursor.description]
                    rows = await cursor.fetchall()
                    
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return []
    
    async def close(self):
        """Close the server."""
        await self.stop() 