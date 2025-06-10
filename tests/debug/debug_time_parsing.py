#!/usr/bin/env python3
"""
Debug script to test time parsing behavior
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from tools.time_parser import parse_timeframe, extract_time_reference
from zoneinfo import ZoneInfo

def test_time_parsing():
    """Test time parsing with current date as June 10, 2025"""
    
    # Set current time as June 10, 2025 (simulating "today")
    test_now = datetime(2025, 6, 10, 15, 30, 0, tzinfo=ZoneInfo("UTC"))
    
    print(f"🕒 Test current time: {test_now}")
    print(f"🕒 Test current date: {test_now.date()}")
    print("=" * 60)
    
    test_queries = [
        "past 2 days",
        "last 2 days", 
        "previous 2 days",
        "give me the highlights of the past 2 days in the buddy groups",
        "2 days ago"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test time reference extraction
        time_ref = extract_time_reference(query)
        print(f"  📝 Extracted time reference: {time_ref}")
        
        # Test timeframe parsing
        try:
            start_dt, end_dt = parse_timeframe(query, now=test_now)
            print(f"  ✅ Parsed timeframe: {start_dt.date()} to {end_dt.date()}")
            print(f"  📅 Expected for 'past 2 days': 2025-06-08 to 2025-06-10")
            
            # Check if parsing is correct
            expected_start = test_now - datetime.timedelta(days=2)
            if abs((start_dt - expected_start).total_seconds()) < 3600:  # Within 1 hour
                print(f"  ✅ CORRECT: Start time matches expected")
            else:
                print(f"  ❌ WRONG: Start time {start_dt.date()} != expected {expected_start.date()}")
                
        except Exception as e:
            print(f"  ❌ Error parsing: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_time_parsing()
