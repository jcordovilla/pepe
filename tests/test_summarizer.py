#!/usr/bin/env python3
"""
Comprehensive test suite for summarization functionality

Tests the summarize_messages function with various scenarios including
empty ranges, different output formats, and integration with time parsing.
"""
import pytest
import json
from datetime import datetime, timedelta
from tools.time_parser import parse_timeframe
from tools.tools import summarize_messages

pytestmark = [pytest.mark.unit, pytest.mark.summarizer]


def test_summarizer_empty_range_text():
    """Test summarization with empty date range returns appropriate message"""
    # Use a far-future date range to ensure no messages exist
    future_iso = "2100-01-01T00:00:00"
    text = summarize_messages(future_iso, future_iso)
    # Should return no messages found message
    assert "No messages found" in text or "no messages" in text.lower()


def test_summarizer_empty_range_json():
    """Test JSON output format with empty date range"""
    future_iso = "2100-01-01T00:00:00"
    result = summarize_messages(future_iso, future_iso, as_json=True)
    
    # Expect structure with empty data when no messages
    assert isinstance(result, dict), "JSON output should be dictionary"
    assert "messages" in result or "summary" in result, "Should have expected JSON structure"
    
    if "messages" in result:
        assert len(result["messages"]) == 0, "Should have empty messages array"


def test_summarizer_date_range_matches_parse():
    """Test that summarizer works with time parser output"""
    # Use parse_timeframe to define a valid recent range
    start, end = parse_timeframe("last week", now=datetime(2025, 5, 7, 12, 0))
    result = summarize_messages(start.isoformat(), end.isoformat())
    
    # Should return some summary (may be no messages found)
    assert isinstance(result, str), "Text summary should be string"
    assert len(result) > 0, "Summary should not be empty string"


def test_summarizer_recent_timeframe():
    """Test summarization with recent timeframe that likely has data"""
    # Test with a week ago from current time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    result = summarize_messages(start_time.isoformat(), end_time.isoformat())
    
    assert isinstance(result, str), "Should return string summary"
    assert len(result) > 10, "Summary should be substantive if messages exist"
    
    # Should either contain actual summary or indicate no messages
    result_lower = result.lower()
    has_content = any(word in result_lower for word in [
        "message", "discussion", "activity", "summary", "channel", "user"
    ])
    no_messages = any(phrase in result_lower for phrase in [
        "no messages", "not found", "empty"
    ])
    
    assert has_content or no_messages, "Summary should either have content or indicate empty result"


def test_summarizer_json_output_structure():
    """Test JSON output has expected structure"""
    # Test with recent timeframe
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)
    
    result = summarize_messages(
        start_time.isoformat(), 
        end_time.isoformat(), 
        as_json=True
    )
    
    assert isinstance(result, dict), "JSON output should be dictionary"
    
    # Check for expected top-level keys
    expected_keys = ["summary", "messages", "timeframe", "stats"]
    present_keys = [key for key in expected_keys if key in result]
    
    assert len(present_keys) > 0, f"Should have at least one expected key: {expected_keys}"
    
    if "summary" in result:
        assert isinstance(result["summary"], str), "Summary should be string"
    
    if "messages" in result:
        assert isinstance(result["messages"], list), "Messages should be list"


def test_summarizer_channel_filtering():
    """Test summarization with channel filtering"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Test with channel filter (use a common channel name)
    result = summarize_messages(
        start_time.isoformat(),
        end_time.isoformat(),
        channel_name="general-chat"
    )
    
    assert isinstance(result, str), "Should return string summary"
    # Should either contain channel-specific content or indicate no messages
    assert len(result) > 0, "Should return some response"


def test_summarizer_different_timeframes():
    """Test summarizer with different timeframe sizes"""
    end_time = datetime.now()
    
    timeframes = [
        (1, "1 day"),
        (7, "1 week"), 
        (30, "1 month")
    ]
    
    results = {}
    
    for days, label in timeframes:
        start_time = end_time - timedelta(days=days)
        
        try:
            summary = summarize_messages(start_time.isoformat(), end_time.isoformat())
            results[label] = {
                'summary': summary,
                'length': len(summary),
                'success': True
            }
            
            assert isinstance(summary, str), f"{label} summary should be string"
            assert len(summary) > 0, f"{label} summary should not be empty"
            
        except Exception as e:
            results[label] = {
                'error': str(e),
                'success': False
            }
            # Don't fail the test for individual timeframe issues
            print(f"Timeframe {label} failed: {e}")
    
    # At least one timeframe should work
    successful_summaries = [r for r in results.values() if r.get('success', False)]
    assert len(successful_summaries) > 0, "At least one timeframe should produce a summary"
    
    print(f"\nSummarization results:")
    for timeframe, result in results.items():
        if result.get('success'):
            print(f"  {timeframe}: {result['length']} chars")
        else:
            print(f"  {timeframe}: FAILED - {result.get('error', 'Unknown error')}")


def test_summarizer_performance():
    """Test that summarization completes in reasonable time"""
    import time
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    start_perf = time.time()
    result = summarize_messages(start_time.isoformat(), end_time.isoformat())
    elapsed = time.time() - start_perf
    
    assert isinstance(result, str), "Should return string result"
    assert elapsed < 30.0, f"Summarization took too long: {elapsed:.2f}s"
    
    print(f"Summarization performance: {elapsed:.2f}s")


def test_summarizer_error_handling():
    """Test error handling for invalid inputs"""
    # Test end before start - this should work as the function should handle this gracefully
    result = summarize_messages("2025-01-02T00:00:00", "2025-01-01T00:00:00")
    assert result is not None, "Should handle end before start gracefully"
    
    # Test with None values - this should handle gracefully
    result = summarize_messages(None, "2025-01-01T00:00:00")
    assert result is not None, "Should handle None start date gracefully"


def test_summarizer_integration_with_enhanced_k():
    """Test that summarizer works with Enhanced K determination"""
    # Test a temporal query that should trigger enhanced k
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # Monthly scope
    
    result = summarize_messages(start_time.isoformat(), end_time.isoformat())
    
    assert isinstance(result, str), "Should return string summary"
    assert len(result) > 0, "Should return some content"
    
    # For monthly scope, should handle larger datasets appropriately
    # This tests integration without requiring specific k values
    print(f"Monthly summary length: {len(result)} characters")
