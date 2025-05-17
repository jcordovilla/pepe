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

@pytest.mark.parametrize("query,expected_keywords", [
    ("Show me discussions about machine learning.", ["machine learning"]),
    ("Find messages containing 'Python'.", ["python"]),
    ("Who has experience in Docker?", ["docker"]),
    ("Show AI messages in #general.", ["general", "ai"]),
    ("Messages from user Alice about deployment.", ["alice", "deployment"]),
])
def test_search_and_filtering(query, expected_keywords):
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="search")
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

@pytest.mark.parametrize("query", [
    "Summarize last week's activity in #dev.",
    "Summarize yesterday in #help as JSON.",
    "Summarize activity in #random."
])
def test_time_scoped_summarization(query):
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="summarize")
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

@pytest.mark.parametrize("query,expected_time_phrases", [
    ("Summarize past 2 days in #dev.", ["2 days"]),
    ("Summarize yesterday.", ["yesterday"]),
    ("Summarize from May 1st to May 5th.", ["may 1", "may 5"]),
])
def test_time_expression_parsing(query, expected_time_phrases):
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="time_parse")
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

@pytest.mark.parametrize("query,should_find", [
    ("Show messages in #announcements.", True),
    ("Show messages in #unknown.", False),
])
def test_channel_name_resolution(query, should_find):
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="channel_resolve")
    if should_find:
        assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"
    else:
        assert ai_eval.startswith("PASS") or "no messages" in ai_eval.lower() or "not found" in ai_eval.lower(), f"AI validation failed: {ai_eval}"

def test_data_availability():
    query = "How many messages are in the database?"
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="data_availability")
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

@pytest.mark.parametrize("query", [
    "Show messages with links.",
    "Show messages with external links."
])
def test_output_formatting_links(query):
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="output_formatting")
    assert ai_eval.startswith("PASS"), f"AI validation failed: {ai_eval}"

def test_error_handling():
    query = "Show messages in #unknown."
    result = get_agent_answer(query)
    ai_eval = ai_validate_response(query, result, functionality="error_handling")
    assert ai_eval.startswith("PASS") or "no messages" in ai_eval.lower() or "not found" in ai_eval.lower(), f"AI validation failed: {ai_eval}"

def test_empty_query():
    with pytest.raises(ValueError):
        get_agent_answer("")
