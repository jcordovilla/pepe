"""
Discord mention resolver for converting user IDs to usernames.
"""
import re
import json
import sqlite3
from typing import Dict, Optional
from core.config import get_config

class MentionResolver:
    """Resolves Discord mentions to actual usernames."""
    
    def __init__(self):
        self._user_cache: Dict[str, str] = {}
        self._load_user_cache()
    
    def _load_user_cache(self):
        """Load user ID to username mapping from database."""
        try:
            config = get_config()
            # Extract SQLite database path from database_url
            db_path = config.database_url.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get unique users from the database
            cursor.execute("SELECT DISTINCT author FROM messages WHERE author IS NOT NULL")
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    author_data = json.loads(row[0])
                    user_id = author_data.get('id')
                    username = author_data.get('username')
                    
                    if user_id and username:
                        self._user_cache[user_id] = username
                        
                except (json.JSONDecodeError, KeyError):
                    continue
                    
            conn.close()
            print(f"Loaded {len(self._user_cache)} users into mention resolver cache")
            
        except Exception as e:
            print(f"Error loading user cache: {e}")
    
    def resolve_mentions(self, content: str) -> str:
        """
        Replace Discord mentions (<@user_id>) with actual usernames.
        
        Args:
            content: Message content with Discord mentions
            
        Returns:
            Content with mentions resolved to usernames
        """
        if not content:
            return content
            
        # Pattern to match Discord user mentions: <@123456789>
        mention_pattern = r'<@(\d+)>'
        
        def replace_mention(match):
            user_id = match.group(1)
            username = self._user_cache.get(user_id)
            
            if username:
                return f"@{username}"
            else:
                # Fallback: keep the user ID but make it more readable
                return f"@User{user_id[-4:]}"  # Use last 4 digits of ID
        
        resolved_content = re.sub(mention_pattern, replace_mention, content)
        
        # Also handle role mentions (<@&role_id>) - convert to @role
        role_pattern = r'<@&(\d+)>'
        resolved_content = re.sub(role_pattern, r'@role\1', resolved_content)
        
        return resolved_content
    
    def get_username(self, user_id: str) -> Optional[str]:
        """Get username for a specific user ID."""
        return self._user_cache.get(user_id)
    
    def refresh_cache(self):
        """Refresh the user cache from database."""
        self._user_cache.clear()
        self._load_user_cache()

# Global instance
_mention_resolver = None

def get_mention_resolver() -> MentionResolver:
    """Get the global mention resolver instance."""
    global _mention_resolver
    if _mention_resolver is None:
        _mention_resolver = MentionResolver()
    return _mention_resolver

def resolve_mentions(content: str) -> str:
    """Convenience function to resolve mentions in content."""
    resolver = get_mention_resolver()
    return resolver.resolve_mentions(content)
