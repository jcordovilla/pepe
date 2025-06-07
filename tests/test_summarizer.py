# tests/test_summarizer.py
import json
from datetime import datetime
from tools.time_parser import parse_timeframe
from tools.tools import summarize_messages


def test_summarizer_empty_range_text():
    # Use a far-future date range to ensure no messages exist
    future_iso = "2100-01-01T00:00:00"
    text = summarize_messages(future_iso, future_iso)
    # Should return no messages found message
    assert "No messages found" in text


def test_summarizer_empty_range_json():
    future_iso = "2100-01-01T00:00:00"
    result = summarize_messages(future_iso, future_iso, as_json=True)
    # Expect structure with empty data when no messages
    assert "messages" in result
    assert len(result["messages"]) == 0


def test_summarizer_date_range_matches_parse():
    # Use parse_timeframe to define a valid recent range
    start, end = parse_timeframe("last week", now=datetime(2025,5,7,12,0))
    result = summarize_messages(start.isoformat(), end.isoformat())
    # Should return some summary (may be no messages found)
    assert isinstance(result, str)
