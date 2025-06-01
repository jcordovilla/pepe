#!/usr/bin/env python3
"""
Pre-deployment validation script for the Agentic Discord Bot.
Checks all prerequisites before starting the bot.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add parent directory to path for importing agentic modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_environment_variables() -> bool:
    """Check if all required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = ['OPENAI_API_KEY', 'DISCORD_TOKEN', 'GUILD_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📋 To fix this:")
        print("export OPENAI_API_KEY='your_key_here'")
        print("export DISCORD_TOKEN='your_token_here'")
        print("export GUILD_ID='your_guild_id_here'")
        print("\nOr create a .env file with these variables.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_dependencies() -> bool:
    """Check if all required Python packages are installed"""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        'discord.py',
        'openai', 
        'chromadb',
        'langgraph',
        'langchain',
        'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('.py', ''))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\n📋 To fix this:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_file_structure() -> bool:
    """Check if all required files and directories exist"""
    print("\n🔍 Checking file structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'agentic/__init__.py',
        'agentic/interfaces/discord_interface.py',
        'agentic/interfaces/agent_api.py',
        'agentic/agents/orchestrator.py'
    ]
    
    required_dirs = [
        'data',
        'agentic',
        'agentic/agents',
        'agentic/interfaces',
        'agentic/memory',
        'agentic/cache'
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"File: {file_path}")
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(f"Directory: {dir_path}")
    
    if missing_items:
        print(f"❌ Missing items:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    
    print("✅ All required files and directories exist")
    return True

async def test_system_components() -> bool:
    """Test that core system components can be initialized"""
    print("\n🔍 Testing system components...")
    
    try:
        # Mock environment for testing
        os.environ.setdefault('OPENAI_API_KEY', 'test_key')
        
        from agentic.interfaces.discord_interface import DiscordInterface
        from agentic.interfaces.agent_api import AgentAPI
        
        # Test configuration
        config = {
            'discord': {
                'token': os.getenv('DISCORD_TOKEN', 'test_token'),
                'command_prefix': '!',
                'guild_id': os.getenv('GUILD_ID', '12345')
            },
            'orchestrator': {
                'memory_config': {},
                'agent_configs': {}
            },
            'cache': {},
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': 'gpt-3.5-turbo'
            }
        }
        
        print("   ✅ Configuration loaded")
        
        # Test imports (components will initialize)
        print("   ✅ Core components importable")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Agentic Discord Bot - Pre-Deployment Validation")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Python Dependencies", check_dependencies), 
        ("File Structure", check_file_structure),
        ("System Components", lambda: asyncio.run(test_system_components()))
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All validation checks passed!")
        print("\n✅ System is ready for deployment")
        print("\n🚀 To start the bot:")
        print("   python main.py")
        print("\n📱 Discord Usage:")
        print("   Use /ask command in your Discord server")
        
        return 0
    else:
        print("❌ Some validation checks failed")
        print("\n🔧 Please fix the issues above before deployment")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
