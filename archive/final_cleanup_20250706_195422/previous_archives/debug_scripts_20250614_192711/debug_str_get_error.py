#!/usr/bin/env python3
"""
Debug the 'str' object has no attribute 'get' error
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def debug_str_get_error():
    """Debug the specific 'str' object has no attribute 'get' error"""
    print("🔍 DEBUGGING 'str' object has no attribute 'get' ERROR")
    print("=" * 60)
    
    # Check one of the files that caused this error
    problem_file = "data/fetched_messages/1353058864810950737_1371894277512364073_messages.json"
    
    try:
        print(f"📄 Analyzing: {problem_file}")
        
        with open(problem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "messages" in data:
            messages = data["messages"]
            print(f"✅ Found {len(messages)} messages")
            
            # Check each message for string objects where dicts are expected
            for i, msg in enumerate(messages[:10]):  # Check first 10 messages
                print(f"\n📨 Message {i+1}:")
                print(f"   Type: {type(msg)}")
                
                if isinstance(msg, str):
                    print(f"   ❌ PROBLEM: Message is a string instead of dict!")
                    print(f"   Content: {msg[:100]}...")
                    break
                elif isinstance(msg, dict):
                    # Check for fields that might be strings when they should be dicts
                    for field_name in ['author', 'reactions', 'mentions', 'attachments']:
                        field_value = msg.get(field_name)
                        print(f"   {field_name}: type={type(field_value)}")
                        
                        if field_name == 'author' and isinstance(field_value, str):
                            print(f"   ❌ PROBLEM: author is string: {field_value}")
                        elif field_name in ['reactions', 'mentions', 'attachments'] and isinstance(field_value, str):
                            print(f"   ❌ PROBLEM: {field_name} is string: {field_value}")
                else:
                    print(f"   ❌ PROBLEM: Message is unexpected type: {type(msg)}")
                    break
                    
        print(f"\n💡 DIAGNOSIS:")
        print("   The error likely occurs when:")
        print("   1. A message itself is a string instead of a dict")
        print("   2. The 'author' field is a string instead of a dict")
        print("   3. Other nested fields are strings instead of objects")
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_str_get_error())
