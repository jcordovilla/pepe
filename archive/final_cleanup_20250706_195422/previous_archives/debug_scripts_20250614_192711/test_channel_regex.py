#!/usr/bin/env python3
"""
Simple test for Discord channel mention regex parsing.
"""

import re

def test_channel_regex():
    """Test the channel regex pattern directly"""
    print("🔍 Testing Discord Channel Mention Regex")
    print("=" * 60)
    
    # The pattern we added
    channel_pattern = r"<#(\d+)>|(?:in|from)\s+(?:the\s+)?#?([\w-]+(?:\s+[\w-]+)*)\s+channel|#([\w🦾🤖🏛🗂❌💻📚🛠❓🌎🏘👋💠-]+)|(?:in|from)\s+([\w-]+-(?:ops|dev|agents|chat|help|support|resources))"
    
    test_cases = [
        {
            "query": "fetch me the last 5 messages from <#1365732945859444767>",
            "expected": "1365732945859444767"
        },
        {
            "query": "what discussions have taken place in <#1371647370911154228> ?",
            "expected": "1371647370911154228"
        },
        {
            "query": "show me content from <#1353448986408779877>",
            "expected": "1353448986408779877"
        },
        {
            "query": "search for AI in <#1353448986408779877>",
            "expected": "1353448986408779877"
        }
    ]
    
    print("🧪 Testing Discord channel mentions:")
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        
        print(f"\n{i}. Query: '{query}'")
        
        # Find all matches
        matches = re.search(channel_pattern, query)
        
        if matches:
            # The Discord mention should be in the first group
            groups = matches.groups()
            channel_id = groups[0] if groups[0] else None
            
            print(f"   Groups found: {groups}")
            print(f"   Channel ID extracted: {channel_id}")
            
            if channel_id == expected:
                print(f"   ✅ SUCCESS: Correctly extracted channel ID")
                success_count += 1
            else:
                print(f"   ❌ FAILED: Expected '{expected}', got '{channel_id}'")
        else:
            print(f"   ❌ FAILED: No channel mention found")
    
    print(f"\n📊 Results: {success_count}/{len(test_cases)} tests passed")
    
    if success_count == len(test_cases):
        print("🎉 All tests passed! Discord channel mentions are working.")
        return True
    else:
        print("❌ Some tests failed. Channel mention parsing needs work.")
        return False

if __name__ == "__main__":
    success = test_channel_regex()
    print(f"\nRegex test {'PASSED' if success else 'FAILED'}")
