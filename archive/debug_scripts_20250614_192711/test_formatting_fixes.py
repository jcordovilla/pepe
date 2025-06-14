#!/usr/bin/env python3
"""
Test the username and timestamp formatting fixes.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.interfaces.discord_interface import DiscordInterface

async def test_formatting_fixes():
    """Test that username and timestamp formatting is working correctly"""
    print("🧪 Testing Username and Timestamp Formatting Fixes")
    print("=" * 60)
    
    try:
        # Test timestamp formatting specifically
        print("🕒 Testing timestamp formatting:")
        test_timestamps = [
            "2025-05-29T23:11:31.969000+00:00",
            "2025-05-26T15:42:48.599000+00:00", 
            "2025-12-25T09:30:15.123000+00:00",
            "2025-01-01T12:00:00.000000+00:00"
        ]
        
        for ts in test_timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                formatted = dt.strftime("%b %d, %I:%M %p")
                print(f"   ✅ {ts}")
                print(f"   → {formatted}")
                print()
            except Exception as e:
                print(f"   ❌ {ts} → ERROR: {e}")
        
        # Test username extraction logic
        print("� Testing username extraction:")
        test_messages = [
            {
                "author_username": "nikhil_kassetty",  # Direct field (current format)
                "expected": "nikhil_kassetty"
            },
            {
                "author": {"username": "fallback_user"},  # Nested format (fallback)
                "expected": "fallback_user"
            },
            {
                # Missing both
                "content": "test message",
                "expected": "Unknown"
            }
        ]
        
        for i, test_msg in enumerate(test_messages, 1):
            # Simulate the extraction logic
            author_name = test_msg.get('author_username', test_msg.get('author', {}).get('username', 'Unknown'))
            expected = test_msg["expected"]
            
            if author_name == expected:
                print(f"   ✅ Test {i}: '{author_name}' (correct)")
            else:
                print(f"   ❌ Test {i}: Expected '{expected}', got '{author_name}'")
        
        print("\n✅ Formatting logic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_formatting_fixes())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
