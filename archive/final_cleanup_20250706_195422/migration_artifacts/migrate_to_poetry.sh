#!/bin/bash

# Pepe Discord Bot - Poetry Migration Script
# Migrates from Python 3.9 + pip to Python 3.11 + Poetry

set -e  # Exit on any error

echo "🚀 Pepe Discord Bot - Migration to Python 3.11 + Poetry"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${GREEN}📋 Step $1: $2${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Step 1: Backup current setup
print_step 1 "Creating backup of current setup"
mkdir -p backup_python39
cp requirements.txt backup_python39/ 2>/dev/null || true
cp -r venv backup_python39/ 2>/dev/null || true
cp .env backup_python39/ 2>/dev/null || true
print_success "Backup created in ./backup_python39/"

# Step 2: Verify Python 3.11 availability
print_step 2 "Verifying Python 3.11 installation"
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version)
    print_success "Found: $PYTHON_VERSION"
else
    print_error "Python 3.11 not found. Install with: brew install python@3.11"
    exit 1
fi

# Step 3: Verify Poetry installation
print_step 3 "Verifying Poetry installation"
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version)
    print_success "Found: $POETRY_VERSION"
else
    print_error "Poetry not found. Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Step 4: Configure Poetry to use Python 3.11
print_step 4 "Configuring Poetry to use Python 3.11"
poetry env use python3.11
print_success "Poetry configured to use Python 3.11"

# Step 5: Install dependencies with Poetry
print_step 5 "Installing dependencies with Poetry"
print_warning "This may take a few minutes..."
poetry install

# Step 6: Verify installation
print_step 6 "Verifying installation"
poetry run python --version
poetry run python -c "import discord, openai, chromadb, langgraph; print('✅ Core dependencies imported successfully')"

# Step 7: Test the system
print_step 7 "Testing the system"
poetry run ./pepe-admin status

print_success "Migration completed successfully!"
echo ""
echo "🎉 Your system is now running:"
echo "   • Python 3.11 (performance boost!)"
echo "   • Poetry for dependency management"
echo "   • All dependencies updated"
echo ""
echo "📋 Next steps:"
echo "   1. Activate Poetry shell: poetry shell"
echo "   2. Test bot: poetry run python main.py"
echo "   3. Run admin commands: poetry run ./pepe-admin status"
echo ""
echo "🗂️  Old environment backed up to: ./backup_python39/"
echo ""
echo "⚡ Enjoy the performance improvements!" 