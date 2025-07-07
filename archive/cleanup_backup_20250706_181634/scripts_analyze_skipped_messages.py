#!/usr/bin/env python3
"""
Analyze the specific messages being skipped to understand what's missing.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def analyze_skipped_messages():
    """Analyze what messages are being skipped and why"""
    print("🔍 ANALYZING SKIPPED MESSAGES")
    print("=" * 60)
    
    message_files = list(Path("data/fetched_messages").glob("*.json"))[:5]  # Check first 5 files
    
    total_messages = 0
    skipped_messages = 0
    skip_reasons = {}
    
    for file_path in message_files:
        print(f"\n📄 Analyzing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data or "messages" not in data:
                print(f"   ⚠️ Invalid file format")
                continue
            
            messages = data["messages"]
            file_total = len(messages)
            file_skipped = 0
            
            for i, msg in enumerate(messages):
                total_messages += 1
                
                # Current filtering logic
                has_content = bool(msg.get("content"))
                has_message_id = bool(msg.get("message_id"))
                
                if not (has_content and has_message_id):
                    file_skipped += 1
                    skipped_messages += 1
                    
                    # Categorize skip reasons
                    if not has_content and not has_message_id:
                        reason = "missing_both_content_and_id"
                    elif not has_content:
                        reason = "missing_content"
                    elif not has_message_id:
                        reason = "missing_message_id"
                    else:
                        reason = "unknown"
                    
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    
                    # Show example of skipped message
                    if file_skipped <= 3:  # Show max 3 examples per file
                        print(f"   ❌ Skipped message {i+1}: {reason}")
                        print(f"      Content: '{str(msg.get('content', 'MISSING'))[:50]}...'")
                        print(f"      Message ID: {msg.get('message_id', 'MISSING')}")
                        print(f"      Available fields: {list(msg.keys())}")
                        print(f"      Message type: {msg.get('type', 'N/A')}")
            
            print(f"   📊 File summary: {file_total - file_skipped}/{file_total} messages kept, {file_skipped} skipped")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n📊 OVERALL ANALYSIS:")
    print("─" * 60)
    print(f"   Total messages analyzed: {total_messages}")
    print(f"   Messages skipped: {skipped_messages} ({skipped_messages/total_messages*100:.1f}%)")
    print(f"   Messages kept: {total_messages - skipped_messages} ({(total_messages-skipped_messages)/total_messages*100:.1f}%)")
    
    print(f"\n❌ SKIP REASONS:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = count / skipped_messages * 100 if skipped_messages > 0 else 0
        print(f"   {reason:<25} : {count:4d} messages ({percentage:5.1f}%)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if skip_reasons.get("missing_content", 0) > 0:
        print(f"   • Empty content messages might be system messages, embeds, or attachments")
        print(f"   • Consider allowing messages with attachments but no text content")
    
    if skip_reasons.get("missing_message_id", 0) > 0:
        print(f"   • Missing message_id is critical - these are likely malformed records")
    
    print(f"   • Suggested fix: Allow messages with message_id + (content OR attachments OR embeds)")
    
if __name__ == "__main__":
    asyncio.run(analyze_skipped_messages())
