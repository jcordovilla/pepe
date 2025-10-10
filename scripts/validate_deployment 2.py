#!/usr/bin/env python3
"""
Pre-deployment validation script for the Agentic Discord Bot.
Checks all prerequisites before starting the bot.
"""

import os
import sys
import asyncio
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for importing agentic modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to the console"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

def print_check_header(title):
    """Print formatted check header"""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print('=' * 60)

def check_environment_variables() -> bool:
    """Check if all required environment variables are set"""
    print_check_header("Environment Variables")
    
    required_vars = ['OPENAI_API_KEY', 'DISCORD_TOKEN', 'GUILD_ID']
    missing_vars = []
    
    print("🔄 Checking environment variables...")
    
    for i, var in enumerate(required_vars):
        time.sleep(0.2)  # Small delay for visual effect
        print_progress_bar(i + 1, len(required_vars), prefix='Progress:', suffix=f'Checking {var}')
        
        if not os.getenv(var):
            missing_vars.append(var)
    
    print()  # New line after progress bar
    
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
    print_check_header("Python Dependencies")
    
    required_packages = [
        'discord.py',
        'openai', 
        'chromadb',
        'langgraph',
        'langchain',
        'sqlalchemy'
    ]
    
    missing_packages = []
    
    print("🔄 Checking Python packages...")
    
    for i, package in enumerate(required_packages):
        time.sleep(0.3)  # Small delay for visual effect
        print_progress_bar(i + 1, len(required_packages), prefix='Progress:', suffix=f'Checking {package}')
        
        try:
            __import__(package.replace('-', '_').replace('.py', ''))
        except ImportError:
            missing_packages.append(package)
    
    print()  # New line after progress bar
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\n📋 To fix this:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_file_structure() -> bool:
    """Check if all required files and directories exist"""
    print_check_header("File Structure")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'agentic/__init__.py',
        'agentic/interfaces/discord_interface.py',
        'agentic/interfaces/agent_api.py',
        'agentic/agents/orchestrator.py'
    ]
    
    required_dirs = [
        'agentic',
        'data',
        'logs',
        'scripts'
    ]
    
    missing_items = []
    all_items = required_files + required_dirs
    
    print("🔄 Checking file structure...")
    
    for i, item in enumerate(all_items):
        time.sleep(0.15)  # Small delay for visual effect
        item_type = "dir" if item in required_dirs else "file"
        print_progress_bar(i + 1, len(all_items), prefix='Progress:', suffix=f'Checking {item_type}: {item}')
        
        if not os.path.exists(item):
            missing_items.append(item)
    
    print()  # New line after progress bar
    
    if missing_items:
        print(f"❌ Missing files/directories: {', '.join(missing_items)}")
        return False
    
    print("✅ All required files and directories exist")
    return True

async def test_system_components() -> bool:
    """Test that core system components can be imported and initialized"""
    print_check_header("System Components")
    
    # Set test environment variables if needed
    if not os.getenv('OPENAI_API_KEY'):
        os.environ.setdefault('OPENAI_API_KEY', 'test_key')
    if not os.getenv('DISCORD_TOKEN'):
        os.environ.setdefault('DISCORD_TOKEN', 'test_token')
    if not os.getenv('GUILD_ID'):
        os.environ.setdefault('GUILD_ID', 'test_guild')
    
    components = [
        ("Discord Interface", "agentic.interfaces.discord_interface"),
        ("Agent API", "agentic.interfaces.agent_api"),
        ("Agent Orchestrator", "agentic.agents.orchestrator"),
        ("Conversation Memory", "agentic.memory.conversation_memory"),
        ("Vector Store", "agentic.vectorstore.persistent_store"),
        ("Smart Cache", "agentic.cache.smart_cache")
    ]
    
    print("🔄 Testing system components...")
    
    failed_components = []
    
    for i, (name, module_path) in enumerate(components):
        time.sleep(0.3)  # Small delay for visual effect
        print_progress_bar(i + 1, len(components), prefix='Progress:', suffix=f'Testing {name}')
        
        try:
            module = __import__(module_path, fromlist=[''])
            # Additional check for key classes if needed
            if hasattr(module, 'DiscordInterface') or hasattr(module, 'AgentAPI') or hasattr(module, 'AgentOrchestrator'):
                pass  # Component loaded successfully
        except Exception as e:
            failed_components.append(f"{name}: {str(e)}")
    
    print()  # New line after progress bar
    
    if failed_components:
        print("❌ Failed to load components:")
        for component in failed_components:
            print(f"   - {component}")
        return False
    
    print("✅ All system components loaded successfully")
    return True

def main():
    """Main validation function"""
    print("🚀 Agentic Discord Bot - Pre-Deployment Validation")
    print("=" * 60)
    print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Python Dependencies", check_dependencies), 
        ("File Structure", check_file_structure),
        ("System Components", lambda: asyncio.run(test_system_components()))
    ]
    
    all_passed = True
    passed_checks = 0
    
    for i, (check_name, check_func) in enumerate(checks, 1):
        print(f"\n📋 Check {i}/{len(checks)}: {check_name}")
        if check_func():
            passed_checks += 1
        else:
            all_passed = False
    
    print(f"\n{'=' * 60}")
    print(f"📊 Validation Summary: {passed_checks}/{len(checks)} checks passed")
    
    if all_passed:
        print("🎉 All validation checks passed!")
        print("\n✅ System is ready for deployment")
        print("\n🚀 To start the bot:")
        print("   python main.py")
        print("\n📱 Discord Usage:")
        print("   Use /ask command in your Discord server")
        print("\n📖 For troubleshooting:")
        print("   python scripts/system_status.py")
        
        return 0
    else:
        print("❌ Some validation checks failed")
        print("\n🔧 Please fix the issues above before deployment")
        print("\n📋 Common fixes:")
        print("   • Set environment variables in .env file")
        print("   • Run: pip install -r requirements.txt")
        print("   • Ensure data/ and logs/ directories exist")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
