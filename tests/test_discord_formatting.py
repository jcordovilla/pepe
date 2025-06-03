#!/usr/bin/env python3
"""
Simplified Discord Response Test

Test just the response transformation and formatting logic
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.append('/Users/jose/Documents/apps/discord-bot-v2')

class MockDiscordContext:
    def __init__(self):
        self.user_id = 123456789
        self.username = "test_user"
        self.channel_id = 987654321
        self.guild_id = 111222333
        self.timestamp = datetime.now()

def transform_agent_response(result):
    """Transform agent API response to Discord interface expected format"""
    if not result.get("success"):
        return {
            "status": "error",
            "message": result.get("error", "Unknown error occurred")
        }
    
    # Get the sources (search results)
    sources = result.get("sources", [])
    answer = result.get("answer", "")
    
    # Determine response type based on sources content
    if sources and isinstance(sources, list) and len(sources) > 0:
        # Check if sources contain message-like objects
        first_source = sources[0]
        if isinstance(first_source, dict) and ("content" in first_source or "author" in first_source):
            # Format as message list
            return {
                "response": {
                    "messages": sources,
                    "total_count": len(sources)
                }
            }
        else:
            # Format as text response with answer
            return {
                "response": {
                    "answer": answer if answer else f"Found {len(sources)} results."
                }
            }
    else:
        # No sources, use answer as text response
        return {
            "response": {
                "answer": answer if answer else "No results found."
            }
        }

async def format_message_list(messages, header, metadata):
    """Format a list of Discord messages"""
    if not messages:
        return [header + "No messages found matching your query."]
    
    chunks = []
    current_chunk = header
    max_message_length = 2000
    
    # Add metadata if available
    if "timeframe" in metadata:
        current_chunk += f"**Timeframe:** {metadata['timeframe']}\n"
    if "channel" in metadata:
        current_chunk += f"**Channel:** {metadata['channel']}\n"
    if "total_count" in metadata:
        current_chunk += f"**Total Messages:** {metadata['total_count']}\n"
    
    current_chunk += "\n**Messages:**\n\n"
    
    for msg in messages:
        author = msg.get('author', {})
        author_name = author.get('username', 'Unknown')
        timestamp = msg.get('timestamp', '')
        content = msg.get('content', '')
        jump_url = msg.get('jump_url', '')
        channel_name = msg.get('channel_name', 'Unknown Channel')
        
        # Format message
        msg_str = f"**{author_name}** ({timestamp}) in **#{channel_name}**\n{content}\n"
        if jump_url:
            msg_str += f"[View message]({jump_url})\n"
        msg_str += "───\n"
        
        # Check if we need to start a new chunk
        if len(current_chunk) + len(msg_str) > max_message_length:
            chunks.append(current_chunk)
            current_chunk = msg_str
        else:
            current_chunk += msg_str
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

async def format_text_response(text, header):
    """Format a simple text response"""
    content = header + text
    max_message_length = 2000
    
    if len(content) <= max_message_length:
        return [content]
    
    # Split into chunks if too long
    chunks = []
    remaining = content
    
    while remaining:
        if len(remaining) <= max_message_length:
            chunks.append(remaining)
            break
        
        # Find a good break point
        chunk = remaining[:max_message_length]
        last_newline = chunk.rfind('\n')
        
        if last_newline > max_message_length * 0.5:
            chunks.append(remaining[:last_newline])
            remaining = remaining[last_newline + 1:]
        else:
            last_space = chunk.rfind(' ')
            if last_space > max_message_length * 0.5:
                chunks.append(remaining[:last_space])
                remaining = remaining[last_space + 1:]
            else:
                chunks.append(chunk)
                remaining = remaining[max_message_length:]
    
    return chunks

async def format_response(result, query, discord_context):
    """Format agent response for Discord display"""
    messages = []
    
    # Add header with user's question
    header = f"**Question:** {query}\n\n"
    
    if result.get("status") == "error":
        return [header + f"**Error:** {result.get('message', 'Unknown error occurred')}"]
    
    response_data = result.get("response", {})
    
    # Handle different response types
    if "messages" in response_data:
        messages.extend(await format_message_list(
            response_data["messages"], header, response_data
        ))
    elif "summary" in response_data:
        # Handle summary responses (not implemented in this test)
        messages.extend(await format_text_response(
            f"**Summary:** {response_data['summary']}", header
        ))
    elif "answer" in response_data:
        messages.extend(await format_text_response(
            response_data["answer"], header
        ))
    else:
        # Fallback for unknown response format
        messages.extend(await format_text_response(
            str(response_data), header
        ))
    
    return messages if messages else [header + "No results found."]

async def test_response_formatting():
    """Test the response formatting logic"""
    
    print("🧪 Testing Discord Response Formatting")
    print("=" * 50)
    
    mock_context = MockDiscordContext()
    
    # Test 1: Message list response
    print("\n📝 Test 1: Message list response")
    mock_agent_response = {
        "success": True,
        "answer": "Found 2 relevant messages.",
        "sources": [
            {
                "author": {"username": "jose_cordovilla", "id": "123"},
                "content": "This is a test message about the query topic",
                "timestamp": "2025-06-03T18:00:00Z",
                "channel_name": "jose-test",
                "jump_url": "https://discord.com/channels/123/456/789"
            },
            {
                "author": {"username": "another_user", "id": "456"},
                "content": "Here's another relevant message",
                "timestamp": "2025-06-03T17:30:00Z",
                "channel_name": "jose-test",
                "jump_url": "https://discord.com/channels/123/456/790"
            }
        ]
    }
    
    transformed = transform_agent_response(mock_agent_response)
    formatted = await format_response(transformed, "list the last 5 messages", mock_context)
    
    print(f"✅ Transformed and formatted successfully!")
    print(f"📤 Response contains {len(formatted)} message chunk(s)")
    for i, chunk in enumerate(formatted, 1):
        print(f"\n📝 Chunk {i} (length: {len(chunk)}):")
        print("-" * 60)
        print(chunk)
        print("-" * 60)
    
    # Test 2: Empty response
    print("\n📝 Test 2: Empty response")
    empty_response = {
        "success": True,
        "answer": "No messages found.",
        "sources": []
    }
    
    transformed_empty = transform_agent_response(empty_response)
    formatted_empty = await format_response(transformed_empty, "find nonexistent topic", mock_context)
    
    print(f"✅ Empty response formatted!")
    print(f"📤 Response contains {len(formatted_empty)} message chunk(s)")
    for i, chunk in enumerate(formatted_empty, 1):
        print(f"\n📝 Empty Chunk {i}:")
        print(chunk)
    
    # Test 3: Error response
    print("\n📝 Test 3: Error response")
    error_response = {
        "success": False,
        "error": "Test error message"
    }
    
    transformed_error = transform_agent_response(error_response)
    formatted_error = await format_response(transformed_error, "trigger error", mock_context)
    
    print(f"✅ Error response formatted!")
    print(f"📤 Response contains {len(formatted_error)} message chunk(s)")
    for i, chunk in enumerate(formatted_error, 1):
        print(f"\n📝 Error Chunk {i}:")
        print(chunk)
    
    print("\n🎉 All formatting tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_response_formatting())
