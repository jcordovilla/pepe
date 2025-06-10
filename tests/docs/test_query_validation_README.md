# Discord Agent System Test Suite

This directory contains comprehensive tests for the Discord Agent System, including unit tests, integration tests, and performance benchmarks.

## 🧪 Test Organization

### Test Files
- **`test_agent_integration.py`** - End-to-end agent functionality tests (26 tests)
- **`test_enhanced_k_determination.py`** - Enhanced K Determination system tests (15 tests)
- **`test_time_parser_comprehensive.py`** - Time parsing functionality (11 tests)
- **`test_summarizer.py`** - Message summarization tests (10 tests)
- **`test_database_integration.py`** - Database operations and FAISS integration
- **`test_performance.py`** - Main performance and load testing
- **`test_utils.py`** - Utility function tests (3 tests)

#### Comprehensive Query Tests
- **`test_queries.json`** - 20 comprehensive test queries covering all agent capabilities
- **`test_query_validation.py`** - Automated validation script for comprehensive query testing
- **Query Coverage**: Server data analysis, feedback summarization, trending topics, Q&A concepts, statistics generation, server structure analysis

#### Integration Tests (`integration/`)
- **`test_local_ai.py`** - Local AI setup validation and testing
- **`test_resource_search.py`** - Resource FAISS index search functionality

#### Performance Tests (`performance/`)
- **`test_embedding_performance.py`** - Embedding model speed and quality benchmarks

### Test Categories (Markers)
- **`unit`** - Fast, isolated unit tests
- **`integration`** - Multi-component integration tests  
- **`performance`** - Performance and load tests (slow)
- **`enhanced_k`** - Enhanced K Determination system tests
- **`database`** - Tests requiring database access
- **`faiss`** - Tests requiring FAISS indices
- **`slow`** - Tests that take significant time

## 🚀 Running Tests

### Quick Test Suite (recommended for development)
```bash
python run_tests.py --suite quick
```

### All Tests (comprehensive)
```bash
python run_tests.py --suite all
```

### Specific Test Categories
```bash
# Run only unit tests
pytest -m "unit and not slow"

# Run Enhanced K tests
pytest tests/test_enhanced_k_determination.py

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```
- Covers parameter validation and edge cases
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.utils`
- **Runtime**: Fast

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
# Integration tests only
pytest -m integration

# Unit tests only  
pytest -m unit

# Specific component tests
pytest -m time_parser
pytest -m summarizer
pytest -m utils
```

### Individual Test Files
```bash
pytest tests/test_agent_integration.py
pytest tests/test_time_parser_comprehensive.py
pytest tests/test_summarizer.py
pytest tests/test_utils.py
```

## Test Organization

The test suite follows a clean 4-file structure:
- **1 integration test file** - Comprehensive end-to-end testing
- **3 focused unit test files** - Component-specific testing

This organization provides:
- Clear separation of concerns
- Comprehensive coverage (47 total tests)
- Easy maintenance and debugging
- Flexible test execution options

## Test Markers

- `integration` - Tests multiple components together
- `unit` - Tests individual functions/classes  
- `time_parser` - Time parsing functionality
- `summarizer` - Message summarization
- `utils` - Utility functions

## Current Status
✅ All 26 integration tests passing  
✅ All unit tests passing  
✅ Total: 47 tests across 4 files  
✅ Well-organized with pytest markers  
✅ Comprehensive coverage of core functionality
