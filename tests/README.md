# Discord Bot Test Suite

## Overview

Streamlined test suite focused on Discord bot core functionality and production readiness.

## Test Structure

### Core Test Files

- **`test_discord_bot_core.py`** - Core functionality tests
  - Vector store operations
  - Agent API functionality  
  - Discord bot commands
  - Forum channel support
  - Sync operations
  - Resource management

- **`test_integration.py`** - Integration tests
  - End-to-end workflows
  - Admin CLI integration
  - Resource management integration
  - Performance testing

- **`run_all_tests.py`** - Unified test runner
  - Quick smoke tests
  - Core functionality tests
  - Integration tests
  - Performance benchmarks
  - Comprehensive reporting

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_all_tests.py

# Quick smoke tests only
python tests/run_all_tests.py --quick

# Core functionality only  
python tests/run_all_tests.py --core

# Integration tests only
python tests/run_all_tests.py --integration

# Performance benchmarks
python tests/run_all_tests.py --performance
```

### Individual Test Suites
```bash
# Core functionality tests
pytest tests/test_discord_bot_core.py -v

# Integration tests
pytest tests/test_integration.py -m integration -v
```

## Test Categories

### 🚀 Quick Tests (< 30 seconds)
- Environment validation
- Module imports
- Basic configuration
- System initialization

### 🔧 Core Tests (1-3 minutes)
- Vector store functionality
- Agent API operations
- Discord command processing
- Data sync operations

### 🔗 Integration Tests (3-5 minutes)
- End-to-end workflows
- Admin CLI operations
- Multi-component interactions
- Resource management

### 📊 Performance Tests (2-5 minutes)
- Response time benchmarks
- Concurrent operation handling
- Memory usage validation
- Bulk processing performance

## Production Readiness

The test suite assesses production readiness based on:

- **95%+ Success Rate**: Production ready 🎉
- **80-94% Success Rate**: Mostly ready, minor issues ⚠️
- **<80% Success Rate**: Critical issues, not ready ❌

## Test Reports

Test reports are automatically generated in `tests/reports/` with:
- Detailed test results
- Performance metrics
- Environment information
- Production readiness assessment

## Legacy Tests

Legacy and redundant tests have been archived in `tests/archive/legacy_tests_*/`
for reference but are not part of the active test suite.

## Requirements

- Python 3.11+
- Poetry environment
- OpenAI API key
- Discord bot token
- All project dependencies installed

## Continuous Integration

This test suite is designed for CI/CD integration with clear exit codes:
- Exit 0: Tests passed (≥80% success rate)
- Exit 1: Tests failed (<80% success rate)
