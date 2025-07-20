# Poetry Environment Management Guide

This guide ensures consistent Poetry usage across the Discord bot project.

## 🎯 Why Poetry?

Poetry provides:
- **Dependency isolation** - Clean virtual environment
- **Reproducible builds** - Lock file ensures consistent versions
- **Easy management** - Simple commands for adding/removing packages
- **Development tools** - Integrated testing, formatting, and linting

## 🚀 Getting Started

### **1. Install Poetry**

```bash
# Install Poetry globally
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### **2. Project Setup**

```bash
# Clone the repository
git clone <repository-url>
cd discord-bot-agentic

# Install dependencies
poetry install

# Verify environment
poetry env info
```

## 🔧 Daily Usage

### **Activating the Environment**

**Option 1: Use `poetry run` for each command**
```bash
# Run any command with Poetry
poetry run python main.py
poetry run ./pepe-admin info
poetry run pytest
```

**Option 2: Activate Poetry shell**
```bash
# Activate the virtual environment
poetry shell

# Now run commands directly
python main.py
./pepe-admin info
pytest
```

### **Verifying Poetry Environment**

```bash
# Check if you're in the Poetry environment
which python
# Should show: /Users/jose/Library/Caches/pypoetry/virtualenvs/pepe-discord-bot-5qtDQwBU-py3.11/bin/python

# Check Poetry environment info
poetry env info
```

## 📋 Essential Commands

### **Discord Bot Operations**

```bash
# Start the bot
poetry run python main.py

# Admin operations
poetry run ./pepe-admin setup
poetry run ./pepe-admin info
poetry run ./pepe-admin sync
poetry run ./pepe-admin test
```

### **Data Management**

```bash
# Fetch Discord messages
poetry run python scripts/discord_message_fetcher.py --incremental

# Index messages for search
poetry run python scripts/index_database_messages.py --incremental

# Resource detection
poetry run python scripts/resource_detector.py --fast-model
```

### **Development**

```bash
# Run tests
poetry run pytest

# Run specific tests
poetry run pytest tests/test_discord_bot_core.py

# Code formatting
poetry run black .
poetry run isort .

# Type checking
poetry run mypy agentic/
```

## 🔄 Dependency Management

### **Adding Dependencies**

```bash
# Add production dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Add with specific version
poetry add "package-name>=1.0.0,<2.0.0"
```

### **Updating Dependencies**

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package-name

# Show outdated packages
poetry show --outdated
```

### **Removing Dependencies**

```bash
# Remove package
poetry remove package-name

# Remove from dev dependencies
poetry remove --group dev package-name
```

## 🧪 Testing with Poetry

### **Running Tests**

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=agentic

# Run specific test file
poetry run pytest tests/test_integration.py

# Run with verbose output
poetry run pytest -v
```

### **Test Configuration**

The project uses pytest configuration in `pyproject.toml`:
- Test discovery: `tests/` directory
- Coverage reporting: `agentic` package
- Minimum version: pytest 6.0+

## 🎨 Code Quality

### **Formatting**

```bash
# Format code with black
poetry run black .

# Sort imports
poetry run isort .

# Check formatting without changes
poetry run black --check .
poetry run isort --check-only .
```

### **Linting**

```bash
# Run flake8
poetry run flake8 agentic/

# Type checking with mypy
poetry run mypy agentic/
```

## 🔍 Troubleshooting

### **Common Issues**

**1. Poetry not found**
```bash
# Reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

**2. Environment not activated**
```bash
# Check current Python path
which python

# Should point to Poetry virtual environment
# If not, use poetry run or poetry shell
```

**3. Dependencies not installed**
```bash
# Reinstall all dependencies
poetry install

# Clear cache and reinstall
poetry cache clear . --all
poetry install
```

**4. Lock file conflicts**
```bash
# Update lock file
poetry lock --no-update

# Or regenerate lock file
rm poetry.lock
poetry install
```

### **Environment Verification**

```bash
# Check Poetry environment
poetry env info

# List installed packages
poetry show

# Check Python version
poetry run python --version
```

## 📚 Best Practices

### **1. Always Use Poetry**

- Never use `pip install` directly
- Always use `poetry add` for new dependencies
- Use `poetry run` or `poetry shell` for all commands

### **2. Keep Dependencies Updated**

```bash
# Regular maintenance
poetry update
poetry show --outdated
```

### **3. Use Development Dependencies**

- Add testing tools to dev dependencies
- Add formatting tools to dev dependencies
- Keep production dependencies minimal

### **4. Commit Lock File**

- Always commit `poetry.lock`
- This ensures reproducible builds
- Never manually edit `poetry.lock`

### **5. Use Virtual Environment**

- Never install packages globally
- Always work within Poetry environment
- Use `poetry shell` for interactive development

## 🚀 Production Deployment

### **Docker Integration**

```dockerfile
# Use Poetry in Docker
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Copy application code
COPY . .

# Run with Poetry
CMD ["poetry", "run", "python", "main.py"]
```

### **CI/CD Integration**

```yaml
# GitHub Actions example
- name: Install Poetry
  run: |
    curl -sSL https://install.python-poetry.org | python3 -

- name: Install dependencies
  run: poetry install

- name: Run tests
  run: poetry run pytest

- name: Format check
  run: poetry run black --check .
```

## 📖 Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry GitHub](https://github.com/python-poetry/poetry)
- [Project pyproject.toml](pyproject.toml)
- [Scripts README](scripts/README.md)

---

**Remember**: Always use Poetry for dependency management and command execution to ensure consistency across all environments! 🎯 