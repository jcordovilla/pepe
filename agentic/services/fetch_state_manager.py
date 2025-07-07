"""
Discord Fetch State Manager
Manages incremental fetching state for Discord message fetcher
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FetchStateManager:
    """
    Manages state for incremental Discord message fetching
    
    Features:
    - Tracks last fetched message timestamp per channel
    - Persistent state storage
    - Safe state updates with backup
    - Channel state cleanup
    """
    
    def __init__(self, state_file: str = "data/processing_markers/fetch_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {}
        self._load_state()
        
        logger.info(f"📊 Fetch state manager initialized: {self.state_file}")
    
    def _load_state(self) -> None:
        """Load state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
                logger.info(f"📖 Loaded state for {len(self._state.get('channels', {}))} channels")
            else:
                self._state = {
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "channels": {}
                }
                logger.info("🆕 Initialized new fetch state")
        except Exception as e:
            logger.error(f"❌ Error loading fetch state: {e}")
            self._state = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "channels": {}
            }
    
    def _save_state(self) -> None:
        """Save state to file with backup"""
        try:
            # Create backup if file exists
            if self.state_file.exists():
                backup_file = self.state_file.with_suffix('.json.backup')
                self.state_file.replace(backup_file)
            
            # Update timestamp
            self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save new state
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved fetch state for {len(self._state.get('channels', {}))} channels")
            
        except Exception as e:
            logger.error(f"❌ Error saving fetch state: {e}")
    
    def get_channel_last_fetch(self, guild_id: str, channel_id: str) -> Optional[datetime]:
        """Get last fetch timestamp for a channel"""
        channel_key = f"{guild_id}_{channel_id}"
        channel_data = self._state.get("channels", {}).get(channel_key)
        
        if channel_data and "last_message_timestamp" in channel_data:
            try:
                return datetime.fromisoformat(channel_data["last_message_timestamp"])
            except ValueError as e:
                logger.warning(f"⚠️ Invalid timestamp format for {channel_key}: {e}")
                return None
        
        return None
    
    def update_channel_state(
        self,
        guild_id: str,
        channel_id: str,
        channel_name: str,
        last_message_timestamp: str,
        message_count: int,
        fetch_mode: str = "incremental"
    ) -> None:
        """Update channel fetch state"""
        channel_key = f"{guild_id}_{channel_id}"
        
        if "channels" not in self._state:
            self._state["channels"] = {}
        
        self._state["channels"][channel_key] = {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "last_message_timestamp": last_message_timestamp,
            "last_fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_message_count": message_count,
            "fetch_mode": fetch_mode
        }
        
        self._save_state()
        logger.debug(f"📝 Updated state for #{channel_name} ({channel_key})")
    
    def get_channel_info(self, guild_id: str, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get full channel state info"""
        channel_key = f"{guild_id}_{channel_id}"
        return self._state.get("channels", {}).get(channel_key)
    
    def remove_channel(self, guild_id: str, channel_id: str) -> None:
        """Remove channel from state (for cleanup)"""
        channel_key = f"{guild_id}_{channel_id}"
        if "channels" in self._state and channel_key in self._state["channels"]:
            del self._state["channels"][channel_key]
            self._save_state()
            logger.info(f"🗑️ Removed channel state: {channel_key}")
    
    def list_channels(self, guild_id: Optional[str] = None) -> Dict[str, Any]:
        """List all channels in state, optionally filtered by guild"""
        channels = self._state.get("channels", {})
        
        if guild_id:
            filtered_channels = {
                k: v for k, v in channels.items()
                if v.get("guild_id") == guild_id
            }
            return filtered_channels
        
        return channels
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        channels = self._state.get("channels", {})
        guild_counts = {}
        
        for channel_data in channels.values():
            guild_id = channel_data.get("guild_id")
            if guild_id:
                guild_counts[guild_id] = guild_counts.get(guild_id, 0) + 1
        
        return {
            "total_channels": len(channels),
            "guilds": guild_counts,
            "last_updated": self._state.get("last_updated"),
            "state_file": str(self.state_file)
        }
    
    def reset_state(self) -> None:
        """Reset all state (use with caution)"""
        self._state = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "channels": {}
        }
        self._save_state()
        logger.warning("🔄 Reset all fetch state")
    
    def cleanup_old_channels(self, days_threshold: int = 30) -> int:
        """Remove channels that haven't been fetched in X days"""
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days_threshold * 24 * 60 * 60)
        channels_to_remove = []
        
        for channel_key, channel_data in self._state.get("channels", {}).items():
            try:
                last_fetch = datetime.fromisoformat(channel_data.get("last_fetch_timestamp", ""))
                if last_fetch.timestamp() < cutoff_date:
                    channels_to_remove.append(channel_key)
            except ValueError:
                # Invalid timestamp, mark for removal
                channels_to_remove.append(channel_key)
        
        # Remove old channels
        for channel_key in channels_to_remove:
            del self._state["channels"][channel_key]
        
        if channels_to_remove:
            self._save_state()
            logger.info(f"🧹 Cleaned up {len(channels_to_remove)} old channel states")
        
        return len(channels_to_remove)
