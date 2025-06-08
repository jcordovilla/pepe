import os
import pytest
from core.agent import get_agent_answer
from core.ai_client import AIClient

# Note: These tests use real database content and agent capabilities aligned with actual data patterns.

pytestmark = pytest.mark.integration

def ai_validate_response(query, response, functionality=None):
    """
    Use local AI to semantically validate the agent's response to the query.
    Returns the model's evaluation string (should start with PASS or FAIL).
    """
    try:
        ai_client = AIClient()
    except Exception as e:
        pytest.skip(f"Local AI not available for validation: {e}")
    # General app context
    app_context = (
        "You are evaluating the output of an AI assistant that helps users query and summarize Discord messages. "
        "The assistant can search for messages by keyword or semantic similarity, summarize messages in a time range, "
        "filter by channel, author, or guild, and return results with metadata such as author, timestamp, and jump URL. "
        "The data consists of real Discord messages from a server, stored in a database and indexed with FAISS for semantic search. "
        "The agent uses LLMs to interpret user queries and generate responses with intelligent routing strategies. "
        "You are to judge if the agent's response is correct, relevant, and complete for the user's query. "
        "Be strict: only reply PASS if the response is clearly correct and complete. Otherwise, reply FAIL and explain why."
    )
    # Add specific context for each functionality
    func_context = ""
    if functionality == "search":
        func_context = (
            "\nFunctionality: Search Discord messages. "
            "The agent should return messages relevant to the query, using both semantic and keyword search. "
            "Results should be filtered by channel, author, or guild if specified. "
            "Returned messages should include metadata such as author, timestamp, and jump URL."
        )
    elif functionality == "summarize":
        func_context = (
            "\nFunctionality: Summarize Discord messages. "
            "The agent should provide a concise and accurate summary of messages within the specified time range and filters. "
            "The summary should reflect the main topics, events, or discussions in the selected scope."
        )
    elif functionality == "time_parse":
        func_context = (
            "\nFunctionality: Time expression parsing. "
            "The agent should correctly interpret natural language time expressions (e.g., 'last week', 'yesterday', 'May 1st to May 5th') "
            "and use them to filter or summarize messages."
        )
    elif functionality == "channel_resolve":
        func_context = (
            "\nFunctionality: Channel name resolution. "
            "The agent should resolve human-friendly channel names to channel IDs and use them to filter messages. "
            "If the channel does not exist, the agent should handle gracefully."
        )
    elif functionality == "data_availability":
        func_context = (
            "\nFunctionality: Data availability. "
            "The agent should report on the number of messages, date range, or other metadata about the database."
        )
    elif functionality == "output_formatting":
        func_context = (
            "\nFunctionality: Output formatting. "
            "The agent should include jump URLs and external links in the results when relevant."
        )
    elif functionality == "error_handling":
        func_context = (
            "\nFunctionality: Error handling. "
            "The agent should handle invalid queries, unknown channels, or empty results gracefully, providing clear feedback."
        )
    prompt = f"""
    {app_context}{func_context}\n\nUser Query: {query}\nAgent Response: {response}\n\nEvaluate if the agent's response correctly and fully answers the user's query. Reply with 'PASS' if it does, or 'FAIL' and a brief explanation if it does not.
    """
    
    # Use local AI for validation
    messages = [{"role": "user", "content": prompt}]
    result = ai_client.chat_completion(messages).strip()
    return result

# Improved suite of 20 end-to-end test prompts aligned with real database content and agent capabilities
@pytest.mark.parametrize("query,expected_behavior", [
    # 1. Meta query - agent routing to data_status (90% confidence)
    ("What data is currently available in the database?", {"functionality": "data_availability"}),
    
    # 2. Search in high-activity channel with real content patterns
    ("Find messages containing 'agent' in #🦾agent-ops.", {"functionality": "search"}),
    
    # 3. Top-K override with real channel and content
    ("Show me the top 4 messages mentioning 'community' in #🌎-community.", {"functionality": "search"}),
    
    # 4. Hybrid search testing (messages + resources) - 85% confidence routing
    ("Search for AI tutorials and documentation about machine learning frameworks.", {"functionality": "search"}),
    
    # 5. Author filter with real active user
    ("List all messages by darkgago in #🏘general-chat.", {"functionality": "search"}),
    
    # 6. Channel-ID filter with real channel and valid date range
    ("Retrieve messages in channel ID 1360692679825948843 (#🏛netarch-general) between 2025-04-01T00:00:00Z and 2025-04-30T23:59:59Z, top 5.", {"functionality": "search"}),
    
    # 7. Relative time summarization in most active channel
    ("Summarize recent activity in #🏘general-chat from the past week.", {"functionality": "summarize"}),
    
    # 8. JSON summarization with real high-activity channel
    ("Summarize messages from 2025-04-01 to 2025-04-30 in #🦾agent-ops as JSON.", {"functionality": "summarize"}),
    
    # 9. Date range summarization in netarch channel
    ("What were the key discussion points in #🏛netarch-general between 2025-04-15 and 2025-04-25?", {"functionality": "summarize"}),
    
    # 10. Skill extraction with real user and channel
    ("Extract technical skills mentioned by manaswita2931 in #❌💻non-coders-learning.", {"functionality": "search"}),
    
    # 11. Semantic search across all channels - messages_only routing (75% confidence)
    ("Find the 3 most semantically similar messages to 'learning programming' across all channels.", {"functionality": "search"}),
    
    # 12. Empty query error handling
    ("", {"functionality": "error_handling", "raises": ValueError, "msg": "Query cannot be empty"}),
    
    # 13. Invalid timeframe error handling
    ("Search messages from 2025-05-26 to 2025-05-20 in #🏘general-chat.", {"functionality": "error_handling", "msg": "End time must be after start time"}),
    
    # 14. Non-existent channel error handling
    ("Search for 'help' in #fake-channel-that-does-not-exist.", {"functionality": "error_handling", "msg": "Unknown channel"}),
    
    # 15. Ambiguous query - should prompt for clarification
    ("Tell me something interesting.", {"functionality": "error_handling", "msg": "Which channel, timeframe, or keyword"}),
    
    # 16. Complex multi-criteria with real channel and user
    ("In #👋introductions, find messages by cristian_72225 from April 2025 and summarize them.", {"functionality": "summarize"}),
    
    # 17. Channel name resolution with embedded emoji
    ("What was discussed about 'onboarding' in the admin-general-chat channel?", {"functionality": "search"}),
    
    # 18. JSON output validation with specific structure
    ("Summarize activity in #❓q-and-a-questions as JSON with summary and key_topics fields.", {"functionality": "summarize", "json_keys": ["summary", "key_topics"]}),
    
    # 19. Recent messages with metadata in high-traffic channel
    ("Give me the 5 most recent messages in #🏘general-chat with jump URLs and timestamps.", {"functionality": "search"}),
    
    # 20. Resource-only query routing (80% confidence) - should access curated resources
    ("Find documentation about Discord bot development and API usage.", {"functionality": "search"}),
])
def test_improved_suite(query, expected_behavior):
    if expected_behavior.get("raises"):
        with pytest.raises(expected_behavior["raises"]):
            get_agent_answer(query)
        return
    result = get_agent_answer(query)
    # Error handling/clarification/fallback
    if expected_behavior["functionality"] == "error_handling":
        if "msg" in expected_behavior:
            assert expected_behavior["msg"].lower() in str(result).lower()
        else:
            ai_eval = ai_validate_response(query, result, functionality="error_handling")
            assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"
        return
    # JSON summary key validation
    if expected_behavior.get("json_keys"):
        assert isinstance(result, dict), "Expected JSON output"
        for key in expected_behavior["json_keys"]:
            assert key in result, f"Missing key '{key}' in JSON summary"
    # Expect empty result
    if expected_behavior.get("expect_empty"):
        assert result == [] or result == {} or not result, "Expected empty result for empty channel search"
        return
    # Standard AI validation
    ai_eval = ai_validate_response(query, result, functionality=expected_behavior["functionality"])
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

def test_empty_query():
    """Test that empty queries are handled appropriately"""
    result = get_agent_answer("")
    # Agent should handle empty queries gracefully rather than raising exception
    assert result is not None
    assert len(str(result)) > 0
    # Should indicate clarification needed or provide help
    result_lower = str(result).lower()
    clarification_indicators = ["clarify", "specify", "help", "what", "how can"]
    assert any(indicator in result_lower for indicator in clarification_indicators), \
        "Empty query should prompt for clarification"

# Additional test for agent routing strategy validation  
@pytest.mark.parametrize("query,expected_routing,min_confidence", [
    # Test agent routing strategies based on query patterns
    ("What's the status of our database?", "data_status", 0.85),  # Meta query
    ("Find Python tutorials", "resources_only", 0.75),  # Resource query
    ("Summarize last week's discussions", "agent_summary", 0.80),  # Summary query
    ("Search for messages about AI ethics", "messages_only", 0.70),  # Semantic query
    ("Find documentation and recent discussions about machine learning", "hybrid_search", 0.80),  # Complex query
])
def test_agent_routing_strategies(query, expected_routing, min_confidence):
    """Test that the agent routes queries to appropriate strategies with expected confidence levels."""
    from core.agent import analyze_query_type
    
    # Test query analysis
    analysis = analyze_query_type(query)
    assert analysis is not None, f"Query analysis failed for: {query}"
    assert 'strategy' in analysis, "Analysis should include strategy"
    assert 'confidence' in analysis, "Analysis should include confidence"
    
    # Verify strategy and confidence
    assert analysis['strategy'] == expected_routing, \
        f"Expected strategy {expected_routing}, got {analysis['strategy']} for query: {query}"
    assert analysis['confidence'] >= min_confidence, \
        f"Confidence {analysis['confidence']:.2f} below minimum {min_confidence} for query: {query}"
    
    # Test that agent actually returns results
    result = get_agent_answer(query)
    assert result is not None
    assert len(str(result)) > 0  # Should return meaningful response

def test_enhanced_k_integration():
    """Test that Enhanced K Determination is integrated with agent queries"""
    from core.agent import _determine_optimal_k
    
    # Test different query types get different k values
    test_cases = [
        ("hi", 5, 30),  # Simple query
        ("weekly digest", 40, 200),  # Temporal query
        ("monthly summary", 200, 1500),  # Larger temporal query  
        ("machine learning tutorials", 15, 50),  # Technical query
        ("quarterly business analysis", 500, 2000)  # Large temporal query
    ]
    
    k_results = {}
    for query, min_k, max_k in test_cases:
        k = _determine_optimal_k(query)
        k_results[query] = k
        
        assert min_k <= k <= max_k, f"Query '{query}' k={k} outside expected range [{min_k}, {max_k}]"
    
    # Test temporal queries get higher k than non-temporal
    assert k_results["weekly digest"] > k_results["hi"], "Temporal queries should get higher k"
    assert k_results["monthly summary"] > k_results["weekly digest"], "Monthly should get higher k than weekly"

def test_context_window_management():
    """Test that the system respects context window limits"""
    # Test with a query that might generate large k
    large_temporal_query = "comprehensive quarterly analysis with detailed monthly breakdowns"
    
    result = get_agent_answer(large_temporal_query)
    assert result is not None, "Large temporal query should return result"
    assert len(str(result)) > 0, "Large temporal query should return meaningful content"
    
    # Result should be within reasonable bounds (not truncated due to context overflow)
    assert len(str(result)) < 50000, "Result should not be excessively large"

def test_preprocessing_field_usage():
    """Test that the system uses preprocessing fields when available"""
    # Query that would benefit from preprocessing fields
    technical_query = "detailed analysis of machine learning algorithms and frameworks"
    
    result = get_agent_answer(technical_query)
    assert result is not None
    assert len(str(result)) > 100, "Technical query should return substantial result"
    
    # Should contain technical terms if preprocessing fields are working
    result_lower = str(result).lower()
    technical_indicators = ["machine", "learning", "algorithm", "framework", "model", "data"]
    found_indicators = sum(1 for indicator in technical_indicators if indicator in result_lower)
    
    assert found_indicators >= 2, f"Technical query should contain technical terms, found: {found_indicators}"
