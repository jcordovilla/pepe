#!/usr/bin/env python3
"""
Debug the task planning and filter extraction process
"""

import asyncio
import sys
import os
import logging
import re
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.append('.')

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_entity_extraction():
    """Test the query analyzer entity extraction"""
    print("🔍 Testing entity extraction...")
    
    # Mock the query analyzer logic
    entity_patterns = {
        "channel": r"#[\w-]+|channel\s+[\w-]+|in\s+([\w-]+)",
        "user": r"@[\w.-]+|by\s+([\w.-]+)|from\s+([\w.-]+)",
        "time_range": r"(last|past|previous)\s+\w+|between\s+[\d-]+\s+and\s+[\d-]+",
        "keyword": r"'([^']+)'|\"([^\"]+)\"|about\s+(\w+)",
        "count": r"top\s+(\d+)|(\d+)\s+results|limit\s+(\d+)",
        "reaction": r"(👍|❤️|😂|👀|🎉|🚀|👏|👌)|(:\w+:)|emoji|reaction"
    }
    
    test_queries = [
        "fetch the 5 last messages in ❌💻non-coders-learning",
        "get messages from #❌💻non-coders-learning",
        "show me messages in non-coders-learning channel",
        "find 3 messages in ❌💻non-coders-learning",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        entities = []
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = {
                    "type": entity_type,
                    "value": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                }
                entities.append(entity)
        
        print(f"  Entities found:")
        for entity in entities:
            print(f"    - {entity['type']}: '{entity['value']}'")
        
        # Test filter building
        filters = {}
        for entity in entities:
            entity_type = entity["type"]
            entity_value = entity["value"]
            
            if entity_type == "channel":
                # Clean channel name
                channel_name = entity_value.lstrip("#").strip()
                filters["channel_name"] = channel_name
            elif entity_type == "count":
                try:
                    filters["k"] = int(entity_value)
                except (ValueError, TypeError):
                    filters["k"] = 5
        
        print(f"  Filters built: {filters}")

def test_channel_name_extraction():
    """Test specific channel name patterns"""
    print("\n🏷️  Testing channel name extraction...")
    
    test_cases = [
        ("❌💻non-coders-learning", "❌💻non-coders-learning"),
        ("#❌💻non-coders-learning", "❌💻non-coders-learning"),
        ("in ❌💻non-coders-learning", "❌💻non-coders-learning"),
        ("messages in ❌💻non-coders-learning", "❌💻non-coders-learning"),
        ("in the ❌💻non-coders-learning channel", "❌💻non-coders-learning"),
    ]
    
    # More comprehensive patterns
    patterns = [
        r"in\s+(❌💻non-coders-learning)",
        r"#(❌💻non-coders-learning)",
        r"(❌💻non-coders-learning)\s+channel",
        r"(❌💻non-coders-learning)(?:\s|$)",
    ]
    
    for text, expected in test_cases:
        print(f"\n  Text: '{text}'")
        print(f"  Expected: '{expected}'")
        
        found = False
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1)
                print(f"  Extracted: '{extracted}' (pattern: {pattern})")
                found = True
                break
        
        if not found:
            print(f"  ❌ No match found")

def test_query_processing_flow():
    """Test the complete query processing flow"""
    print("\n🔄 Testing complete query processing flow...")
    
    query = "fetch the 5 last messages in ❌💻non-coders-learning"
    
    # Step 1: Entity extraction
    print(f"1. Query: {query}")
    
    # Extract channel with better pattern
    channel_pattern = r"in\s+(❌💻[\w-]+)|#(❌💻[\w-]+)|(❌💻[\w-]+)"
    channel_match = re.search(channel_pattern, query)
    
    if channel_match:
        # Get the first non-None group
        channel_name = next((g for g in channel_match.groups() if g), None)
        print(f"2. Channel extracted: '{channel_name}'")
    else:
        channel_name = None
        print(f"2. ❌ No channel extracted")
    
    # Extract count
    count_pattern = r"(\d+)\s+(last|messages)|top\s+(\d+)|limit\s+(\d+)"
    count_match = re.search(count_pattern, query)
    
    if count_match:
        # Get the first non-None group that's a number
        count = next((int(g) for g in count_match.groups() if g and g.isdigit()), 5)
        print(f"3. Count extracted: {count}")
    else:
        count = 5
        print(f"3. Default count: {count}")
    
    # Step 3: Build filters
    filters = {}
    if channel_name:
        filters["channel_name"] = channel_name
    
    print(f"4. Filters built: {filters}")
    
    # Step 4: Simulate search parameters  
    search_params = {
        "query": query,
        "k": count,
        "filters": filters
    }
    
    print(f"5. Search parameters: {search_params}")

def main():
    """Main debug function"""
    print("🐛 Debugging task planning and filter extraction...")
    print("=" * 60)
    
    test_entity_extraction()
    test_channel_name_extraction()
    test_query_processing_flow()
    
    print("\n" + "=" * 60)
    print("🏁 Debug complete")

if __name__ == "__main__":
    main()
