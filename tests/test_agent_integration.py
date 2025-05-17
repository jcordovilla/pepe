import os
import openai
import pytest
from core.agent import get_agent_answer

# Note: These tests assume a test database with known data is available.
# If not, they should be adapted to use fixtures/mocks or run in a controlled environment.

def ai_validate_response(query, response, functionality=None):
    """
    Use OpenAI to semantically validate the agent's response to the query.
    Returns the model's evaluation string (should start with PASS or FAIL).
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gpt_model = os.getenv("GPT_MODEL", "gpt-4-turbo")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping AI validation.")
    # General app context
    app_context = (
        "You are evaluating the output of an AI assistant that helps users query and summarize Discord messages. "
        "The assistant can search for messages by keyword or semantic similarity, summarize messages in a time range, "
        "filter by channel, author, or guild, and return results with metadata such as author, timestamp, and jump URL. "
        "The data consists of real Discord messages from a server, stored in a database and indexed with FAISS for semantic search. "
        "The agent uses LLMs to interpret user queries and generate responses. "
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
    completion = openai.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )
    result = completion.choices[0].message.content.strip()
    return result

# Balanced suite of 20 end-to-end test prompts
@pytest.mark.parametrize("query,expected_behavior", [
    # 1. Validate data availability
    ("What data is currently cached?", {"functionality": "data_availability"}),
    # 2. Basic keyword search
    ("Find messages containing ‘AI ethics’ in #📚ai-philosophy-ethics.", {"functionality": "search"}),
    # 3. Top-K override
    ("Show me the top 4 messages mentioning ‘welcome’ in #📝welcome-rules.", {"functionality": "search"}),
    # 4. Hybrid keyword + semantic
    ("Search for ‘use cases’ with keyword ‘roundtable’ in #genai-use-case-roundtable, returning the top 3 results.", {"functionality": "search"}),
    # 5. Author filter
    ("List all messages by cristian_72225 in #👋introductions.", {"functionality": "search"}),
    # 6. Channel-ID filter with ISO times
    ("Retrieve messages in channel ID 1365732945859444767 (📢announcements-admin) between 2025-04-20T00:00:00Z and 2025-04-25T23:59:59Z, top 5.", {"functionality": "search"}),
    # 7. Plain-text summarization (relative)
    ("Summarize messages from last weekend in #🏘general-chat.", {"functionality": "summarize"}),
    # 8. JSON summarization (absolute)
    ("Summarize messages from 2025-04-01 to 2025-04-30 in #🛠ai-practical-applications as JSON.", {"functionality": "summarize"}),
    # 9. Plain-text summarization (absolute)
    ("What were the key discussion points in #🤖intro-to-agentic-ai between 2025-04-22 and 2025-04-24?", {"functionality": "summarize"}),
    # 10. Skill-term extraction
    ("Extract all skills mentioned by darkgago in #🛠ai-practical-applications over the past month.", {"functionality": "search"}),
    # 11. Semantic-only reranking
    ("Find the 3 most semantically similar messages to ‘invite link restrictions’ across all channels.", {"functionality": "search"}),
    # 12. Empty-query guard
    ("", {"functionality": "error_handling", "raises": ValueError, "msg": "Query cannot be empty"}),
    # 13. Invalid timeframe
    ("Search messages from 2025-04-26 to 2025-04-23 in #🏘general-chat.", {"functionality": "error_handling", "msg": "End time must be after start time"}),
    # 14. Unknown channel
    ("Search for ‘bug’ in #nonexistent-channel.", {"functionality": "error_handling", "msg": "Unknown channel"}),
    # 15. Fallback/clarification prompt
    ("Tell me something interesting.", {"functionality": "error_handling", "msg": "Which channel, timeframe, or keyword"}),
    # 16. Combined multi-criteria
    ("In #📥feedback-submissions, find the top 2 messages about ‘suggestions’ by laura.neder between last Friday and yesterday, then summarize them.", {"functionality": "summarize"}),
    # 17. Implicit channel name parsing
    ("What was discussed about ‘help’ on 2025-04-18 in the discord-help channel?", {"functionality": "search"}),
    # 18. JSON summary key validation
    ("Summarize this week in #❓q-and-a-questions as JSON and ensure the output has both `summary` and `note` fields.", {"functionality": "summarize", "json_keys": ["summary", "note"]}),
    # 19. Jump-URL accuracy
    ("Give me the 5 most recent messages in #🏘general-chat with jump URLs.", {"functionality": "search"}),
    # 20. Empty-channel search
    ("Retrieve messages in channel ID 1364250555467300976 (📩midweek-request) between 2025-04-20 and 2025-04-25.", {"functionality": "search", "expect_empty": True}),
])
def test_balanced_suite(query, expected_behavior):
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
    with pytest.raises(ValueError):
        get_agent_answer("")
