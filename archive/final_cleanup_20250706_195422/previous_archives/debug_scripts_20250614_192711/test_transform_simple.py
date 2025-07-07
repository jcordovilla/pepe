# Test the response transformation logic directly
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

# Test cases
print("Test 1: Response with message sources")
mock_response = {
    "success": True,
    "answer": "I found 2 relevant messages.",
    "sources": [
        {
            "author": {"username": "test_user"},
            "content": "Test message 1",
            "timestamp": "2025-06-03T18:00:00Z",
            "channel_name": "test-channel"
        },
        {
            "author": {"username": "test_user2"},
            "content": "Test message 2", 
            "timestamp": "2025-06-03T18:01:00Z",
            "channel_name": "test-channel"
        }
    ]
}

result = transform_agent_response(mock_response)
print(result)
print()

print("Test 2: Response with empty sources")
mock_empty = {
    "success": True,
    "answer": "No messages found.",
    "sources": []
}

result_empty = transform_agent_response(mock_empty)
print(result_empty)
print()

print("Test 3: Error response")
mock_error = {
    "success": False,
    "error": "Query failed"
}

result_error = transform_agent_response(mock_error)
print(result_error)
