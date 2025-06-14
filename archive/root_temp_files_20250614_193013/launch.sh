#!/bin/bash
# Launcher script for Agentic Discord Bot
#
# This script provides easy commands to run different components of the agentic system.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize environment
init_env() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/data"
    
    # Activate virtual environment
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        log_success "Virtual environment activated"
    else
        log_warning "Virtual environment not found at $VENV_PATH"
        log_info "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    fi
}

# Main commands
case "$1" in
    "bot"|"discord")
        log_info "🤖 Starting Agentic Discord Bot..."
        init_env
        python3 main.py
        ;;
    
    "streamlit"|"web")
        log_info "🌐 Starting Streamlit Web Interface..."
        init_env
        python3 -m streamlit run agentic/interfaces/streamlit_interface.py
        ;;
    
    "test")
        log_info "🧪 Running tests..."
        init_env
        python3 -c "
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

async def test_framework():
    from agentic.interfaces.agent_api import AgentAPI
    
    config = {
        'orchestrator': {'memory_config': {'db_path': 'data/conversation_memory.db'}},
        'vectorstore': {'persist_directory': 'data/vectorstore', 'collection_name': 'test_messages'},
        'cache': {'redis_url': 'redis://localhost:6379'}
    }
    
    api = AgentAPI(config)
    result = await api.query('Hello, test the system', 'test_user')
    print('✅ Test successful:', 'success' in result and result['success'])

asyncio.run(test_framework())
        "
        ;;
    
    "setup")
        log_info "⚙️ Setting up project..."
        
        # Create virtual environment if it doesn't exist
        if [ ! -d "$VENV_PATH" ]; then
            log_info "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate and install dependencies
        source "$VENV_PATH/bin/activate"
        log_info "Installing dependencies..."
        pip install -r requirements.txt
        
        # Create necessary directories
        mkdir -p data logs
        
        log_success "Setup complete!"
        log_info "Next steps:"
        log_info "1. Update .env file with your API keys"
        log_info "2. Run: ./launch.sh bot"
        ;;
    
    "clean")
        log_warning "🧹 Cleaning up temporary files..."
        rm -rf __pycache__/
        rm -rf .pytest_cache/
        rm -f *.log
        find . -name "*.pyc" -delete
        log_success "Cleanup complete!"
        ;;
    
    "status")
        log_info "📊 System Status..."
        init_env
        python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('🔧 Environment Variables:')
print(f'  OPENAI_API_KEY: {\"✅ Set\" if os.getenv(\"OPENAI_API_KEY\") else \"❌ Missing\"}')
print(f'  DISCORD_TOKEN: {\"✅ Set\" if os.getenv(\"DISCORD_TOKEN\") else \"❌ Missing\"}')
print(f'  GUILD_ID: {\"✅ Set\" if os.getenv(\"GUILD_ID\") else \"❌ Missing\"}')

print('\\n📁 Directories:')
print(f'  data/: {\"✅ Exists\" if os.path.exists(\"data\") else \"❌ Missing\"}')
print(f'  logs/: {\"✅ Exists\" if os.path.exists(\"logs\") else \"❌ Missing\"}')
print(f'  venv/: {\"✅ Exists\" if os.path.exists(\"venv\") else \"❌ Missing\"}')
        "
        ;;
    
    "pipeline"|"update")
        log_info "🔄 Running Discord message database update pipeline..."
        init_env
        python3 run_pipeline.py
        ;;
    
    "help"|*)
        echo "🚀 Agentic Discord Bot Launcher"
        echo ""
        echo "Usage: ./launch.sh [command]"
        echo ""
        echo "Commands:"
        echo "  bot, discord    Start the Discord bot"
        echo "  streamlit, web  Start the Streamlit web interface"
        echo "  test           Run framework tests"
        echo "  setup          Initial project setup"
        echo "  pipeline, update Update Discord messages database"
        echo "  clean          Clean temporary files"
        echo "  status         Show system status"
        echo "  help           Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./launch.sh setup     # First time setup"
        echo "  ./launch.sh bot       # Start Discord bot"
        echo "  ./launch.sh streamlit # Start web interface"
        echo "  ./launch.sh pipeline  # Update Discord messages database"
        ;;
esac
