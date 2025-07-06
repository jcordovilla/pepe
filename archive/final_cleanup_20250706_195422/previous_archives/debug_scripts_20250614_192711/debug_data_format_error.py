#!/usr/bin/env python3
"""
Debug script to identify the specific data format issue causing 'int' has no len() error
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def debug_specific_file_error():
    """Debug the specific file causing the error"""
    print("🔍 DEBUGGING MESSAGE FILE DATA FORMAT ISSUE")
    print("=" * 60)
    
    # Check the specific problematic file
    problem_file = "data/fetched_messages/1353058864810950737_1366389827221717002_messages.json"
    
    try:
        print(f"📄 Analyzing: {problem_file}")
        
        if not Path(problem_file).exists():
            print(f"❌ File not found: {problem_file}")
            return
            
        with open(problem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"📋 File structure:")
        print(f"   Type: {type(data)}")
        print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if "messages" in data:
            messages = data["messages"]
            print(f"   Messages type: {type(messages)}")
            print(f"   Messages count: {len(messages) if hasattr(messages, '__len__') else 'No len() method'}")
            
            if hasattr(messages, '__len__') and len(messages) > 0:
                print(f"\n🔍 Analyzing first message structure:")
                first_msg = messages[0]
                print(f"   Message type: {type(first_msg)}")
                
                if isinstance(first_msg, dict):
                    print(f"   Message keys: {list(first_msg.keys())}")
                    
                    # Check specific fields that might cause issues
                    problematic_fields = []
                    
                    for key, value in first_msg.items():
                        if key in ['reactions', 'mentions', 'attachments', 'embeds']:
                            print(f"   {key}: type={type(value)}, value={value}")
                            if isinstance(value, int) and key in ['reactions', 'mentions', 'attachments']:
                                problematic_fields.append(key)
                        elif key == 'author':
                            print(f"   author: type={type(value)}")
                            if isinstance(value, dict):
                                print(f"      author keys: {list(value.keys())}")
                            else:
                                problematic_fields.append('author')
                    
                    if problematic_fields:
                        print(f"\n❌ PROBLEMATIC FIELDS FOUND: {problematic_fields}")
                        print("   These fields are integers but code expects lists/dicts")
                    else:
                        print(f"\n✅ No obvious problematic fields detected")
                        
                    # Show the exact content that might be causing issues
                    print(f"\n📋 Reactions field analysis:")
                    reactions = first_msg.get('reactions')
                    print(f"   reactions = {reactions} (type: {type(reactions)})")
                    
                    print(f"\n📋 Mentions field analysis:")
                    mentions = first_msg.get('mentions')  
                    print(f"   mentions = {mentions} (type: {type(mentions)})")
                    
                    print(f"\n📋 Attachments field analysis:")
                    attachments = first_msg.get('attachments')
                    print(f"   attachments = {attachments} (type: {type(attachments)})")
                    
                else:
                    print(f"   ❌ First message is not a dict: {first_msg}")
            else:
                print(f"   ❌ Messages list is empty or doesn't have len()")
        else:
            print(f"   ❌ No 'messages' key found in data")
            
        # Test the exact code that's failing
        print(f"\n🧪 TESTING VECTOR STORE ADD_MESSAGES CODE:")
        
        if "messages" in data and data["messages"]:
            msg = data["messages"][0]
            
            try:
                # This is the exact code from add_messages that's failing
                content = msg.get("content", "").strip()
                print(f"   ✅ content extraction: OK")
                
                reactions = msg.get("reactions", [])
                print(f"   reactions = {reactions} (type: {type(reactions)})")
                
                # This line might be failing:
                total_reactions = sum(r.get("count", 0) for r in reactions) if reactions else 0
                print(f"   ❌ This line likely fails if reactions is int")
                
            except Exception as e:
                print(f"   ❌ Error in vector store code simulation: {e}")
                print(f"   💡 This confirms the data format issue")
                
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_specific_file_error())
