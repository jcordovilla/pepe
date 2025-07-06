#!/usr/bin/env python3
"""
Test the fixed channel resolution directly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic.services.channel_resolver import ChannelResolver

async def test_fixed_channel_resolution():
    """Test the newly added channel resolution method"""
    print("🔧 Testing Fixed Channel Resolution")
    print("=" * 60)
    
    try:
        # Initialize channel resolver
        resolver = ChannelResolver()
        
        # Test the specific channel IDs from Discord queries
        test_channel_ids = [
            "1371647370911154228",  # Should resolve to ❌💻non-coders
            "1353448986408779877",  # Should resolve to 📚ai-philosophy-ethics
            "1365732945859444767",  # Should return None (not in vector store)
        ]
        
        print("🔍 Testing channel ID resolution:")
        for channel_id in test_channel_ids:
            resolved_name = resolver.resolve_channel_id_to_name(channel_id)
            if resolved_name:
                print(f"   ✅ {channel_id} -> '{resolved_name}'")
            else:
                print(f"   ❌ {channel_id} -> Not found")
        
        print("\n✅ Channel resolution test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_channel_resolution())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
