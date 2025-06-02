#!/usr/bin/env python3
"""
Test Analytics Structure and Integration

This test validates that the analytics system is properly structured
and integrated without requiring API keys or external dependencies.
"""

import sys
import traceback
from pathlib import Path

def test_analytics_imports():
    """Test that all analytics modules can be imported"""
    print("🧪 Testing Analytics Module Imports...")
    
    try:
        # Test analytics package imports
        from agentic.analytics import (
            QueryAnswerRepository, 
            PerformanceMonitor, 
            ValidationSystem, 
            AnalyticsDashboard
        )
        print("✅ Analytics package imports successful")
        
        # Test that classes can be instantiated with minimal config
        config = {
            "db_path": ":memory:",  # Use in-memory SQLite
            "monitoring_interval": 60,
            "thresholds": {}
        }
        
        # Test QueryAnswerRepository
        query_repo = QueryAnswerRepository(config)
        print("✅ QueryAnswerRepository instantiated successfully")
        
        # Test PerformanceMonitor  
        perf_monitor = PerformanceMonitor(config)
        print("✅ PerformanceMonitor instantiated successfully")
        
        # Test ValidationSystem
        validation_system = ValidationSystem(config)
        print("✅ ValidationSystem instantiated successfully")
        
        # Test AnalyticsDashboard
        dashboard = AnalyticsDashboard(config)
        print("✅ AnalyticsDashboard instantiated successfully")
        
        # Test component linking
        dashboard.set_components(query_repo, perf_monitor, validation_system)
        print("✅ Component linking successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics imports test failed: {e}")
        traceback.print_exc()
        return False

def test_interface_structure():
    """Test that interface classes have required analytics methods"""
    print("\n🧪 Testing Interface Analytics Structure...")
    
    try:
        # Test Discord Interface
        from agentic.interfaces.discord_interface import DiscordInterface
        
        discord_methods = [
            "get_analytics_dashboard",
            "get_performance_analytics"
        ]
        
        for method in discord_methods:
            if hasattr(DiscordInterface, method):
                print(f"✅ DiscordInterface.{method} found")
            else:
                print(f"❌ DiscordInterface.{method} missing")
                return False
        
        # Test Streamlit Interface
        from agentic.interfaces.streamlit_interface import StreamlitInterface
        
        streamlit_methods = [
            "render_analytics_page",
            "_render_overview_analytics",
            "_render_performance_analytics", 
            "_render_user_analytics",
            "_render_quality_analytics"
        ]
        
        for method in streamlit_methods:
            if hasattr(StreamlitInterface, method):
                print(f"✅ StreamlitInterface.{method} found")
            else:
                print(f"❌ StreamlitInterface.{method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Interface structure test failed: {e}")
        traceback.print_exc()
        return False

def test_analytics_data_structures():
    """Test analytics data structures"""
    print("\n🧪 Testing Analytics Data Structures...")
    
    try:
        from agentic.analytics.query_answer_repository import QueryMetrics, ValidationResult
        
        # Test QueryMetrics - provide all required constructor args
        metrics = QueryMetrics(
            response_time=1.5,
            agents_used=["agent1", "agent2"],
            tokens_used=100,
            cache_hit=True,
            success=True,
            error_message=None
        )
        print("✅ QueryMetrics data structure working")
        
        # Test ValidationResult - this is a proper dataclass
        validation = ValidationResult(
            is_valid=True,
            quality_score=4.42,
            relevance_score=4.5,
            completeness_score=4.0,
            accuracy_score=4.8,
            issues=[],
            suggestions=["Good response"]
        )
        print("✅ ValidationResult data structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Analytics Structure Tests")
    print("=" * 50)
    
    results = {
        "Analytics Imports": test_analytics_imports(),
        "Interface Structure": test_interface_structure(), 
        "Data Structures": test_analytics_data_structures()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All analytics structure tests passed!")
        print("✅ Analytics system is properly integrated and ready for use")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
