#!/usr/bin/env python3
"""
Test script to analyze the new Discord message fields in the database.
This script demonstrates how to query and analyze the enhanced message data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.db import SessionLocal, Message
from collections import Counter
import json

def analyze_enhanced_fields():
    """Analyze the new Discord message fields in the database."""
    
    session = SessionLocal()
    
    try:
        # Get total message count
        total_messages = session.query(Message).count()
        print(f"📊 Total messages in database: {total_messages}")
        
        if total_messages == 0:
            print("❌ No messages found. Run fetch_messages.py first.")
            return
        
        # Sample a few messages to check field availability
        sample_messages = session.query(Message).limit(100).all()
        
        # Analyze field coverage
        field_stats = {
            'edited_messages': 0,
            'messages_with_embeds': 0,
            'messages_with_attachments': 0,
            'messages_with_reactions': 0,
            'pinned_messages': 0,
            'tts_messages': 0,
            'reply_messages': 0,
            'messages_with_stickers': 0,
            'system_messages': 0,
            'bot_messages': 0
        }
        
        message_types = Counter()
        flag_usage = Counter()
        
        for msg in sample_messages:
            # Check for edited messages
            if msg.edited_at:
                field_stats['edited_messages'] += 1
            
            # Check for embeds
            if msg.embeds and len(msg.embeds) > 0:
                field_stats['messages_with_embeds'] += 1
            
            # Check for attachments
            if msg.attachments and len(msg.attachments) > 0:
                field_stats['messages_with_attachments'] += 1
            
            # Check for reactions
            if msg.reactions and len(msg.reactions) > 0:
                field_stats['messages_with_reactions'] += 1
            
            # Check for pinned messages
            if msg.pinned:
                field_stats['pinned_messages'] += 1
            
            # Check for TTS messages
            if msg.tts:
                field_stats['tts_messages'] += 1
            
            # Check for replies
            if msg.reference:
                field_stats['reply_messages'] += 1
            
            # Check for stickers
            if msg.stickers and len(msg.stickers) > 0:
                field_stats['messages_with_stickers'] += 1
            
            # Check message types
            if msg.type:
                message_types[msg.type] += 1
                if msg.type != 'MessageType.default':
                    field_stats['system_messages'] += 1
            
            # Check for bot messages
            if msg.webhook_id or msg.application_id:
                field_stats['bot_messages'] += 1
            
            # Track flag usage
            if msg.flags and msg.flags > 0:
                flag_usage[msg.flags] += 1
        
        print(f"\n📈 Enhanced Field Analysis (sample of {len(sample_messages)} messages):")
        print("=" * 60)
        
        for field, count in field_stats.items():
            percentage = (count / len(sample_messages)) * 100 if len(sample_messages) > 0 else 0
            print(f"  {field.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n🏷️ Message Types Found:")
        for msg_type, count in message_types.most_common():
            percentage = (count / len(sample_messages)) * 100
            print(f"  {msg_type}: {count} ({percentage:.1f}%)")
        
        if flag_usage:
            print(f"\n🚩 Message Flags Usage:")
            for flag, count in flag_usage.most_common():
                percentage = (count / len(sample_messages)) * 100
                print(f"  Flag {flag}: {count} ({percentage:.1f}%)")
        
        # Show example of rich content if available
        rich_message = session.query(Message).filter(
            (Message.embeds.isnot(None)) | 
            (Message.attachments.isnot(None)) |
            (Message.reference.isnot(None))
        ).first()
        
        if rich_message:
            print(f"\n🎨 Example of Rich Content Message:")
            print("=" * 60)
            print(f"Message ID: {rich_message.message_id}")
            print(f"Channel: #{rich_message.channel_name}")
            print(f"Author: {rich_message.author.get('username', 'Unknown')}")
            print(f"Content: {rich_message.content[:100]}...")
            
            if rich_message.embeds:
                print(f"Embeds: {len(rich_message.embeds)} embed(s)")
            if rich_message.attachments:
                print(f"Attachments: {len(rich_message.attachments)} file(s)")
            if rich_message.reference:
                print(f"Reply to: Message {rich_message.reference.get('message_id', 'Unknown')}")
            if rich_message.edited_at:
                print(f"Last edited: {rich_message.edited_at}")
        
        print(f"\n✅ Enhanced Discord message field analysis complete!")
        print(f"📖 See docs/discord_message_fields.md for detailed field documentation")
        
    except Exception as e:
        print(f"❌ Error analyzing enhanced fields: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.close()

if __name__ == "__main__":
    analyze_enhanced_fields()
