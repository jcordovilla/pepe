#!/usr/bin/env python3
"""
Debug script to investigate the specific error in message processing.
"""

import asyncio
import json
import traceback
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_message_file_error():
    """Debug the specific file causing the 'object of type 'int' has no len()' error"""
    print("🔍 DEBUGGING MESSAGE FILE PROCESSING ERROR")
    print("=" * 60)
    
    # Load the problematic file
    problem_file = "data/fetched_messages/1353058864810950737_1366389827221717002_messages.json"
    
    try:
        print(f"📁 Loading file: {problem_file}")
        with open(problem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"📊 File structure: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            
            if "messages" in data:
                messages = data["messages"]
                print(f"   Messages: {len(messages)} items")
                print(f"   Messages type: {type(messages)}")
                
                if messages:
                    # Check first message structure
                    first_msg = messages[0]
                    print(f"\n📨 First message structure:")
                    print(f"   Type: {type(first_msg)}")
                    
                    if isinstance(first_msg, dict):
                        print(f"   Keys: {list(first_msg.keys())}")
                        
                        # Check for problematic fields
                        problematic_fields = []
                        for key, value in first_msg.items():
                            if hasattr(value, '__len__') and not isinstance(value, (str, list, dict)):
                                problematic_fields.append((key, type(value), value))
                        
                        if problematic_fields:
                            print(f"   ⚠️ Problematic fields:")
                            for field, field_type, value in problematic_fields:
                                print(f"      {field}: {field_type} = {value}")
                        
                        # Check specific fields that might cause len() issues
                        potential_issues = []
                        for key, value in first_msg.items():
                            try:
                                if key in ['reactions', 'mentions', 'attachments', 'embeds']:
                                    if value is not None:
                                        test_len = len(value)
                                        print(f"   ✅ {key}: {type(value)} (len={test_len})")
                                    else:
                                        print(f"   ℹ️ {key}: None")
                            except Exception as e:
                                potential_issues.append((key, type(value), value, str(e)))
                                print(f"   ❌ {key}: {type(value)} = {value} -> ERROR: {e}")
                        
                        if potential_issues:
                            print(f"\n🎯 FOUND THE ISSUE:")
                            for field, field_type, value, error in potential_issues:
                                print(f"   Field '{field}' has type {field_type} with value {value}")
                                print(f"   Error: {error}")
                        
                        # Now simulate the problematic code path
                        print(f"\n🧪 SIMULATING MESSAGE PROCESSING...")
                        
                        try:
                            # This is the code from add_messages that might be failing
                            reactions = first_msg.get("reactions", [])
                            print(f"   reactions = {reactions} (type: {type(reactions)})")
                            
                            if reactions:
                                try:
                                    total_reactions = sum(r.get("count", 0) for r in reactions)
                                    print(f"   ✅ total_reactions calculation successful: {total_reactions}")
                                except Exception as e:
                                    print(f"   ❌ total_reactions calculation failed: {e}")
                                
                                try:
                                    reaction_emojis = [r.get("emoji", "") for r in reactions]
                                    print(f"   ✅ reaction_emojis calculation successful: {reaction_emojis}")
                                except Exception as e:
                                    print(f"   ❌ reaction_emojis calculation failed: {e}")
                            
                            # Check mentions
                            mentions = first_msg.get("mentions", [])
                            print(f"   mentions = {mentions} (type: {type(mentions)})")
                            
                            if mentions:
                                try:
                                    mentioned_user_ids = [str(m.get("id", "")) for m in mentions]
                                    print(f"   ✅ mentioned_user_ids calculation successful: {mentioned_user_ids}")
                                except Exception as e:
                                    print(f"   ❌ mentioned_user_ids calculation failed: {e}")
                            
                            # Check attachments
                            attachments = first_msg.get("attachments", [])
                            print(f"   attachments = {attachments} (type: {type(attachments)})")
                            
                            if attachments:
                                try:
                                    attachment_urls = [a.get("url", "") for a in attachments]
                                    print(f"   ✅ attachment_urls calculation successful: {attachment_urls}")
                                except Exception as e:
                                    print(f"   ❌ attachment_urls calculation failed: {e}")
                            
                            # Check embeds
                            embeds = first_msg.get("embeds", [])
                            print(f"   embeds = {embeds} (type: {type(embeds)})")
                            
                            try:
                                embed_count = len(embeds)
                                print(f"   ✅ embed_count calculation successful: {embed_count}")
                            except Exception as e:
                                print(f"   ❌ embed_count calculation failed: {e}")
                                print(f"      This is likely the source of the error!")
                            
                            # Check content
                            content = first_msg.get("content", "")
                            print(f"   content type: {type(content)}")
                            
                            try:
                                content_len = len(content)
                                print(f"   ✅ content length calculation successful: {content_len}")
                            except Exception as e:
                                print(f"   ❌ content length calculation failed: {e}")
                            
                        except Exception as e:
                            print(f"   ❌ Error in message processing simulation: {e}")
                            traceback.print_exc()
                    
                    else:
                        print(f"   ❌ First message is not a dictionary: {type(first_msg)}")
                
                else:
                    print(f"   ❌ Messages list is empty")
            else:
                print(f"   ❌ No 'messages' key found")
        
        else:
            print(f"   ❌ File is not a dictionary: {type(data)}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {problem_file}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_message_file_error()
