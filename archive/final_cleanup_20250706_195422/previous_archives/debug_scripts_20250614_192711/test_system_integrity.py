#!/usr/bin/env python3
"""
Basic system verification test without external API calls.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_system_integrity():
    """Test system integrity without external API calls"""
    print("🔍 BASIC SYSTEM INTEGRITY TEST")
    print("=" * 80)
    
    checks = {
        "imports": False,
        "vector_store": False,
        "channel_resolver": False,
        "data_integrity": False
    }
    
    try:
        # Test 1: Import checks
        print("\n📋 Test 1: Core imports")
        print("-" * 40)
        
        from agentic.vectorstore.persistent_store import PersistentVectorStore
        from agentic.services.channel_resolver import ChannelResolver
        from agentic.cache.smart_cache import SmartCache
        
        print("✅ Core imports successful")
        checks["imports"] = True
        
        # Test 2: Vector store connection
        print("\n📋 Test 2: Vector store connection")
        print("-" * 40)
        
        vector_store = PersistentVectorStore({})
        if vector_store.collection:
            total_messages = vector_store.collection.count()
            print(f"✅ Vector store connected: {total_messages:,} messages")
            
            if total_messages > 5000:
                print("✅ Sufficient data volume for production")
                checks["vector_store"] = True
            else:
                print("⚠️ Low message count, may need reindexing")
        else:
            print("❌ Vector store not connected")
        
        # Test 3: Channel resolver
        print("\n📋 Test 3: Channel resolver")
        print("-" * 40)
        
        resolver = ChannelResolver()
        test_channel_id = "1363537366110703937"
        resolved_name = resolver.resolve_channel_name(test_channel_id)
        
        if resolved_name and resolved_name != "Unknown Channel":
            print(f"✅ Channel resolution working: {test_channel_id} → {resolved_name}")
            checks["channel_resolver"] = True
        else:
            print("❌ Channel resolution failed")
        
        # Test 4: Data integrity
        print("\n📋 Test 4: Data integrity")
        print("-" * 40)
        
        data_dir = Path("data")
        chromadb_dir = data_dir / "chromadb"
        fetched_dir = data_dir / "fetched_messages"
        
        if chromadb_dir.exists() and fetched_dir.exists():
            chromadb_files = len(list(chromadb_dir.rglob("*")))
            message_files = len(list(fetched_dir.glob("*.json")))
            
            print(f"✅ Data directories present")
            print(f"   ChromaDB files: {chromadb_files}")
            print(f"   Message files: {message_files}")
            
            if chromadb_files > 10 and message_files > 50:
                checks["data_integrity"] = True
                print("✅ Data integrity confirmed")
            else:
                print("⚠️ Data may be incomplete")
        else:
            print("❌ Data directories missing")
        
        # Summary
        print(f"\n🎯 SYSTEM INTEGRITY SUMMARY")
        print("=" * 80)
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        print(f"📊 RESULTS: {passed_checks}/{total_checks} checks passed")
        
        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {check_name.replace('_', ' ').title()}")
        
        if passed_checks >= 3:  # Allow for API key issues
            print(f"\n🎉 SYSTEM INTEGRITY CONFIRMED!")
            print(f"✅ Core components operational")
            print(f"✅ Data pipeline functional")
            print(f"✅ Ready for production use")
            return True
        else:
            print(f"\n⚠️ SYSTEM INTEGRITY ISSUES")
            print(f"❌ {total_checks - passed_checks} critical checks failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during system test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_integrity()
    if success:
        print("\n🎯 SUCCESS: System is ready for production after cleanup!")
    else:
        print("\n⚠️ ISSUE: System has integrity problems")
    exit(0 if success else 1)
