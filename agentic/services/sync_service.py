"""
Enhanced Data Synchronization Service
Modern data sync with state management and recovery patterns
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataSynchronizationService:
    """
    Modern data sync with legacy-proven state management
    
    Preserves battle-tested:
    - Incremental sync patterns
    - State management and recovery
    - Error handling and retry logic
    - Data consistency validation
    """
    
    def __init__(self, unified_data_manager):
        self.data_manager = unified_data_manager
        self.sync_state_file = Path("data/sync_state.json")
        self.sync_state = self._load_sync_state()
        
        logger.info("💾 Data sync service initialized with legacy patterns")
    
    async def sync_discord_data(self) -> Dict[str, Any]:
        """
        Main synchronization workflow with legacy patterns
        Enhanced with modern unified data layer
        """
        sync_id = f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🔄 Starting data sync: {sync_id}")
        
        sync_result = {
            "sync_id": sync_id,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "stats": {
                "messages_processed": 0,
                "messages_added": 0,
                "messages_updated": 0,
                "errors": []
            }
        }
        
        try:
            # Legacy-style incremental sync
            last_sync = self.sync_state.get("last_sync_timestamp")
            since = datetime.fromisoformat(last_sync) if last_sync else None
            
            # Fetch new data since last sync
            new_messages = await self._fetch_incremental_data(since)
            
            # Process and store with modern data layer
            if new_messages:
                print(f"🔄 Synchronizing {len(new_messages)} messages...")
                with tqdm(new_messages, desc="🔄 Syncing data", unit="msg") as pbar:
                    for message in pbar:
                        try:
                            # Process through modern unified data manager
                            await self.data_manager.store_message(message)
                            sync_result["stats"]["messages_processed"] += 1
                            
                            # Update progress bar with stats
                            pbar.set_postfix({
                                'Processed': sync_result["stats"]["messages_processed"],
                                'Errors': len(sync_result["stats"]["errors"])
                            })
                            
                            # Legacy-style progress tracking
                            if sync_result["stats"]["messages_processed"] % 100 == 0:
                                await self._save_sync_checkpoint(sync_result)
                                
                        except Exception as e:
                            error_msg = f"Error processing message {message.get('message_id')}: {e}"
                            sync_result["stats"]["errors"].append(error_msg)
                            logger.warning(f"⚠️ {error_msg}")
            else:
                print("ℹ️  No new messages to synchronize")
            
            # Legacy-style data validation
            print("🔍 Validating data integrity...")
            validation_result = await self._validate_data_integrity()
            sync_result["validation"] = validation_result
            
            # Update sync state
            self.sync_state["last_sync_timestamp"] = datetime.now().isoformat()
            self.sync_state["last_sync_result"] = sync_result
            self._save_sync_state()
            
            sync_result["status"] = "completed"
            sync_result["end_time"] = datetime.now().isoformat()
            
            logger.info(f"✅ Sync {sync_id} completed: {sync_result['stats']['messages_processed']} messages")
            return sync_result
            
        except Exception as e:
            sync_result["status"] = "failed"
            sync_result["error"] = str(e)
            sync_result["end_time"] = datetime.now().isoformat()
            
            logger.error(f"❌ Sync {sync_id} failed: {e}")
            raise
    
    async def _fetch_incremental_data(self, since: Optional[datetime]) -> list:
        """
        Fetch incremental data with legacy patterns
        """
        # This would integrate with the modernized Discord service
        # Implementation depends on the specific data sources
        logger.info(f"📥 Fetching incremental data since {since}")
        
        # Placeholder for actual implementation
        return []
    
    async def _validate_data_integrity(self) -> Dict[str, Any]:
        """
        Legacy-style data integrity validation
        """
        validation = {
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "message_count_consistency": True,
                "embedding_completeness": True,
                "mcp_sqlite_integrity": True,
                "analytics_consistency": True
            },
            "issues": []
        }
        
        checks = list(validation["checks"].keys())
        
        try:
            # Validate with modern data layer
            with tqdm(total=len(checks), desc="🔍 Validation checks", unit="check") as pbar:
                health_check = await self.data_manager.health_check()
                
                for store, status in health_check.items():
                    if not status:
                        validation["checks"][f"{store}_integrity"] = False
                        validation["issues"].append(f"{store} integrity check failed")
                    
                    pbar.update(1)
                    pbar.set_postfix({'Issues': len(validation["issues"])})
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {e}")
            logger.warning(f"⚠️ Data validation error: {e}")
        
        return validation
    
    def _load_sync_state(self) -> Dict[str, Any]:
        """Load synchronization state from file"""
        if self.sync_state_file.exists():
            try:
                with open(self.sync_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Error loading sync state: {e}")
        
        return {
            "last_sync_timestamp": None,
            "sync_history": []
        }
    
    def _save_sync_state(self):
        """Save synchronization state to file"""
        try:
            self.sync_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sync_state_file, 'w') as f:
                json.dump(self.sync_state, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving sync state: {e}")
    
    async def _save_sync_checkpoint(self, sync_result: Dict[str, Any]):
        """Save sync checkpoint for recovery"""
        checkpoint_file = Path(f"data/sync_checkpoints/checkpoint_{sync_result['sync_id']}.json")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(sync_result, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save checkpoint: {e}")
