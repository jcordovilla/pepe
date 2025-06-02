#!/usr/bin/env python3
"""
Test Analytics Integration

This script tests the comprehensive analytics integration across all interfaces.
"""

import asyncio
import sys
import os
from datetime import datetime
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic.interfaces.agent_api import AgentAPI
from agentic.interfaces.discord_interface import DiscordInterface


async def test_agent_api_analytics():
    """Test AgentAPI analytics integration"""
    print("🧪 Testing AgentAPI Analytics Integration...")
    
    try:
        # Initialize AgentAPI with analytics
        config = {
            "orchestrator": {},
            "vector_store": {},
            "memory": {"db_path": "data/conversation_memory.db"},
            "pipeline": {},
            "analytics": {
                "db_path": "data/analytics.db",
                "performance_monitoring": True,
                "validation_enabled": True
            }
        }
        
        agent_api = AgentAPI(config)
        
        # Test analytics components initialization
        assert agent_api.query_repository is not None, "QueryRepository not initialized"
        assert agent_api.performance_monitor is not None, "PerformanceMonitor not initialized"
        assert agent_api.validation_system is not None, "ValidationSystem not initialized"
        assert agent_api.analytics_dashboard is not None, "AnalyticsDashboard not initialized"
        
        print("✅ Analytics components initialized successfully")
        
        # Test query with analytics recording
        test_query = "What is the purpose of this system?"
        result = await agent_api.query(
            query=test_query,
            user_id="test_user_123",
            context={
                "platform": "test",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        assert result is not None, "Query result is None"
        assert "status" in result, "Query result missing status"
        
        print("✅ Query with analytics recording successful")
        
        # Test analytics dashboard generation
        dashboard_data = await agent_api.analytics_dashboard.generate_overview_dashboard(
            hours_back=24,
            platform="test"
        )
        
        assert dashboard_data is not None, "Dashboard data is None"
        assert "timestamp" in dashboard_data, "Dashboard missing timestamp"
        
        print("✅ Analytics dashboard generation successful")
        
        await agent_api.close()
        print("✅ AgentAPI analytics integration test completed successfully")
        
    except Exception as e:
        print(f"❌ AgentAPI analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def test_discord_analytics():
    """Test Discord interface analytics integration"""
    print("\n🧪 Testing Discord Analytics Integration...")
    
    try:
        # Initialize Discord interface
        config = {
            "orchestrator": {},
            "vector_store": {},
            "memory": {"db_path": "data/conversation_memory.db"},
            "pipeline": {},
            "analytics": {
                "db_path": "data/analytics.db",
                "performance_monitoring": True,
                "validation_enabled": True
            }
        }
        
        agent_api = AgentAPI(config)
        discord_interface = DiscordInterface(agent_api=agent_api)
        
        # Test analytics dashboard access
        dashboard_data = await discord_interface.get_analytics_dashboard()
        
        assert dashboard_data is not None, "Dashboard data is None"
        print("✅ Discord analytics dashboard access successful")
        
        # Test performance analytics
        performance_data = await discord_interface.get_performance_analytics(hours_back=24)
        
        assert performance_data is not None, "Performance data is None"
        print("✅ Discord performance analytics successful")
        
        await agent_api.close()
        print("✅ Discord analytics integration test completed successfully")
        
    except Exception as e:
        print(f"❌ Discord analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_streamlit_analytics_structure():
    """Test Streamlit interface analytics structure (without Streamlit dependencies)"""
    print("\n🧪 Testing Streamlit Analytics Structure...")
    
    try:
        # Import and check structure
        from agentic.interfaces.streamlit_interface import StreamlitInterface
        
        # Check that analytics methods exist
        methods = [
            'render_analytics_page',
            '_render_overview_analytics',
            '_render_performance_analytics',
            '_render_user_analytics',
            '_render_quality_analytics',
            '_render_fallback_overview',
            '_render_fallback_performance',
            '_render_fallback_users'
        ]
        
        for method in methods:
            assert hasattr(StreamlitInterface, method), f"Method {method} not found"
        
        print("✅ All required analytics methods found in StreamlitInterface")
        print("✅ Streamlit analytics structure test completed successfully")
        
    except Exception as e:
        print(f"❌ Streamlit analytics structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Run all analytics integration tests"""
    print("🚀 Starting Analytics Integration Tests")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    results = []
    
    # Test AgentAPI analytics
    results.append(await test_agent_api_analytics())
    
    # Test Discord analytics  
    results.append(await test_discord_analytics())
    
    # Test Streamlit analytics structure
    results.append(test_streamlit_analytics_structure())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   AgentAPI Analytics: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"   Discord Analytics:  {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"   Streamlit Structure: {'✅ PASS' if results[2] else '❌ FAIL'}")
    
    if all(results):
        print("\n🎉 All analytics integration tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
