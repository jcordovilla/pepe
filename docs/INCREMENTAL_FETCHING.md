# 🔄 Incremental Discord Message Fetching

## Overview

The Discord message fetcher now supports **incremental fetching**, which means it will only fetch new messages since the last fetch operation, dramatically reducing fetch time and API usage.

## How It Works

### Checkpoint System
- **Checkpoint File**: `data/fetching_checkpoint.json`
- **Per-Channel Tracking**: Each channel and thread maintains its own last message ID
- **Automatic Updates**: Checkpoints are updated after each successful fetch

### Fetch Modes

#### 1. **Incremental Fetch (Default)**
```bash
# Fetch only new messages since last run
poetry run ./pepe-admin sync

# Or directly
poetry run python scripts/discord_message_fetcher.py
```

**Behavior:**
- ✅ Loads existing checkpoint
- ✅ Fetches messages after the last known message ID
- ✅ Updates checkpoint with new last message ID
- ✅ Shows "🔄 Incremental fetch" status

#### 2. **Full Fetch (Force)**
```bash
# Force full fetch, ignoring checkpoints
poetry run ./pepe-admin sync --full

# Or directly
poetry run python scripts/discord_message_fetcher.py --full
```

**Behavior:**
- 🗑️ Deletes existing checkpoint
- 📥 Fetches all messages from the beginning
- 💾 Creates new checkpoint
- ✅ Shows "📥 Full fetch" status

#### 3. **Reset Checkpoint**
```bash
# Reset checkpoint and start fresh
poetry run python scripts/discord_message_fetcher.py --reset-checkpoint
```

**Behavior:**
- 🗑️ Deletes existing checkpoint
- 📥 Performs full fetch on next run
- ℹ️ Useful for troubleshooting

## Checkpoint File Structure

```json
{
  "channel_checkpoints": {
    "123456789": "987654321",
    "987654321": "555666777"
  },
  "forum_checkpoints": {
    "111222333": "444555666",
    "777888999": "000111222"
  },
  "last_updated": "2025-01-20T16:30:45.123456",
  "total_messages_fetched": 15000
}
```

## Benefits

### 🚀 **Performance**
- **First Run**: Full fetch (may take 10-30 minutes)
- **Subsequent Runs**: Incremental fetch (usually 1-5 minutes)
- **API Efficiency**: Only requests new messages

### 💰 **Cost Savings**
- **Reduced API Calls**: Only fetches new content
- **Lower Bandwidth**: Minimal data transfer
- **Faster Processing**: Less data to process

### 🔄 **Convenience**
- **Automatic**: No manual intervention needed
- **Reliable**: Checkpoint system ensures no data loss
- **Flexible**: Can force full fetch when needed

## Usage Examples

### Daily Sync (Recommended)
```bash
# Quick daily sync - only new messages
poetry run ./pepe-admin sync
```

### Weekly Full Sync
```bash
# Complete refresh - all messages
poetry run ./pepe-admin sync --full
```

### Troubleshooting
```bash
# Reset checkpoint if issues occur
poetry run python scripts/discord_message_fetcher.py --reset-checkpoint
poetry run ./pepe-admin sync
```

### Partial Operations
```bash
# Only fetch new messages (no indexing)
poetry run ./pepe-admin sync --fetch-only

# Only index existing messages (no fetching)
poetry run ./pepe-admin sync --index-only
```

## Progress Indicators

### Incremental Fetch
```
🔄 Incremental fetch from message 12345678...
   ✅ 45 new messages from #general
   ✅ 12 new messages from #announcements
```

### Full Fetch
```
📥 Full fetch (no checkpoint found)...
📥 #general: 100%|██████████| 1500/1500 [02:30<00:00, 10.0msgs/s]
   ✅ 1,500 messages from #general
```

## Troubleshooting

### Checkpoint Issues
```bash
# Check checkpoint file
cat data/fetching_checkpoint.json

# Reset if corrupted
rm data/fetching_checkpoint.json
poetry run ./pepe-admin sync
```

### Force Full Sync
```bash
# When incremental isn't working
poetry run ./pepe-admin sync --full
```

### Database Verification
```bash
# Check message counts
sqlite3 data/discord_messages.db "SELECT COUNT(*) FROM messages;"
sqlite3 data/discord_messages.db "SELECT MAX(timestamp) FROM messages;"
```

## Migration from Old System

If you're upgrading from the old non-incremental system:

1. **First Run**: Will perform full fetch and create checkpoint
2. **Subsequent Runs**: Will use incremental fetching
3. **No Data Loss**: All existing messages are preserved

## Best Practices

### ✅ **Recommended**
- Run incremental sync daily: `poetry run ./pepe-admin sync`
- Use full sync weekly: `poetry run ./pepe-admin sync --full`
- Monitor checkpoint file for issues
- Keep checkpoint file in backups

### ❌ **Avoid**
- Manually editing checkpoint file
- Deleting checkpoint without backup
- Running full sync unnecessarily

## Technical Details

### Checkpoint Management
- **Automatic Creation**: Created on first successful fetch
- **Per-Channel Tracking**: Each channel has independent checkpoint
- **Thread Support**: Forum threads have separate checkpoints
- **Error Recovery**: Failed fetches don't update checkpoints

### API Optimization
- **After Parameter**: Uses Discord's `after` parameter for efficiency
- **Batch Processing**: Processes messages in batches of 100
- **Rate Limiting**: Respects Discord API rate limits
- **Error Handling**: Graceful handling of API errors

### Database Integration
- **INSERT OR IGNORE**: Prevents duplicate messages
- **Transaction Safety**: Batch inserts with rollback on error
- **Index Optimization**: Uses existing database indices
- **NULL Handling**: Proper handling of NULL values

## Monitoring

### Check Sync Status
```bash
# View system information
poetry run ./pepe-admin info

# Check database statistics
sqlite3 data/discord_messages.db "SELECT COUNT(*) as total, MAX(timestamp) as latest FROM messages;"
```

### Log Analysis
```bash
# Check recent fetch logs
tail -f logs/discord_fetch.log

# Monitor checkpoint updates
ls -la data/fetching_checkpoint.json
```

---

**🎉 Incremental fetching is now the default behavior!** 

Your Discord bot will automatically use incremental fetching for faster, more efficient sync operations while maintaining full data integrity. 