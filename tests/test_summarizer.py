# tests/test_summarizer.py
import json
from datetime import datetime
from time_parser import parse_timeframe
from tools import summarize_messages_in_range


def test_summarizer_empty_range_text():
    # Use a far-future date range to ensure no messages exist
    future_iso = "2100-01-01T00:00:00"
    text = summarize_messages_in_range(future_iso, future_iso)
    # Header must mention the correct range
    assert text.startswith("📅 Messages from 2100-01-01 to 2100-01-01")
    # Should have no channel sections after header
    lines = text.splitlines()
    assert len(lines) == 1 or all(not line.startswith("**") for line in lines[1:])


def test_summarizer_empty_range_json():
    future_iso = "2100-01-01T00:00:00"
    j = summarize_messages_in_range(future_iso, future_iso, output_format="json")
    data = json.loads(j)
    # Expect an empty dict when no messages
    assert data == {}


def test_summarizer_date_range_matches_parse():
    # Use parse_timeframe to define a valid recent range
    start, end = parse_timeframe("last week", now=datetime(2025,5,7,12,0))
    text = summarize_messages_in_range(start.isoformat(), end.isoformat())
    # Header must match parse_timeframe output
    expected_header = f"📅 Messages from {start.date()} to {end.date()}"
    assert text.splitlines()[0] == expected_header
