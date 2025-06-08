# Query Logging System

The Discord bot now has comprehensive logging of all user interactions from both interfaces:

## 📁 Log Locations

### 1. Simple Text Logs (Easy to Search)
- **Location**: `logs/simple_logs/`
- **Format**: `queries_simple_YYYYMMDD.txt`
- **Content**: Human-readable format with full queries and responses
- **Best for**: Quick searching and reading

### 2. Structured JSON Logs (Analytics)
- **Location**: `logs/query_logs/`
- **Format**: `queries_YYYYMMDD.jsonl`
- **Content**: Structured data with metadata, performance metrics
- **Best for**: Analytics and detailed analysis

### 3. Database Storage
- **Location**: SQLite database via QueryLog model
- **Content**: Full structured data with relationships
- **Best for**: Complex queries and reporting

## 🔍 Searching Logs

### Quick Search Script
```bash
# Search for specific terms
python search_logs.py "AI"
python search_logs.py "machine learning"

# Filter by interface
python search_logs.py --interface discord "help"
python search_logs.py --interface streamlit "summary"

# Filter by user
python search_logs.py --user "john" 

# Search specific date
python search_logs.py --date 20250609 "error"

# List available log files
python search_logs.py --list-files
```

### Manual File Search
```bash
# Search simple logs with grep
grep -r "search term" logs/simple_logs/

# Search JSON logs
grep "search term" logs/query_logs/*.jsonl
```

## 📊 What Gets Logged

### From Discord (`/pepe` command):
- ✅ User ID and username
- ✅ Guild and channel information  
- ✅ Full query text
- ✅ Complete response
- ✅ Processing time and performance metrics
- ✅ Query analysis (strategy, confidence)
- ✅ Error details if any

### From Streamlit Web Interface:
- ✅ Session-based user tracking
- ✅ Query text and parameters
- ✅ Complete response
- ✅ Channel filters applied
- ✅ Processing time and performance metrics
- ✅ Error details if any

## 📈 Log File Examples

### Simple Text Log Entry:
```
=== QUERY LOG ENTRY ===
Timestamp: 2025-06-09T15:30:45.123456
Interface: DISCORD
User: john_doe (ID: 123456789)
Guild ID: 987654321
Channel: general-chat

QUERY:
Show me messages about AI from last week

RESPONSE:
Here are the AI-related messages from last week:
[Response content...]

==================================================
```

### JSON Log Entry:
```json
{
  "id": 1,
  "timestamp": "2025-06-09T15:30:45.123456",
  "user_id": "123456789",
  "username": "john_doe",
  "query_text": "Show me messages about AI from last week",
  "query_type": "semantic_search",
  "routing_strategy": "messages_only",
  "confidence_score": 0.85,
  "response_status": "success",
  "processing_time_ms": 1250,
  "search_results_count": 15,
  "is_successful": true
}
```

## 🛠️ Log Management

### Daily Log Rotation
- New files created daily automatically
- Format: `queries_simple_YYYYMMDD.txt` and `queries_YYYYMMDD.jsonl`

### Log Cleanup (Optional)
```bash
# Remove logs older than 30 days
find logs/simple_logs/ -name "*.txt" -mtime +30 -delete
find logs/query_logs/ -name "*.jsonl" -mtime +30 -delete
```

### Monitoring Log Growth
```bash
# Check log sizes
du -sh logs/simple_logs/
du -sh logs/query_logs/

# Count entries in today's logs
wc -l logs/query_logs/queries_$(date +%Y%m%d).jsonl
```

## 🔧 Configuration

The logging system is automatically enabled for both interfaces. No additional configuration needed.

### Environment Variables (Optional)
- Logs are written to `logs/` directory relative to the project root
- JSON logs include full query and response text
- Simple logs are human-readable and easily searchable

## 📋 Common Use Cases

### Find errors:
```bash
python search_logs.py "ERROR"
grep -r "ERROR" logs/simple_logs/
```

### Track user behavior:
```bash
python search_logs.py --user "username"
```

### Monitor specific topics:
```bash
python search_logs.py "machine learning"
python search_logs.py "troubleshoot"
```

### Performance analysis:
```bash
# Check JSON logs for processing times > 5 seconds
grep -E '"processing_time_ms": [5-9][0-9]{3}' logs/query_logs/*.jsonl
```

---

All user interactions are now automatically logged to easily searchable files! 🎉
