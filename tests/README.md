# Test Suite Documentation

## Overview
This test suite validates the Discord bot's functionality with comprehensive coverage across all major components.

## Test Files

### `test_agent_integration.py` - Integration Tests
- **26 comprehensive integration tests** 
- Tests the complete agent workflow including query routing, validation, and response generation
- Covers error handling, date validation, channel resolution, and AI response validation
- **Marker**: `@pytest.mark.integration`
- **Runtime**: Moderate (uses real database and AI validation)

### `test_time_parser_comprehensive.py` - Time Parser Tests  
- **15 extensive time parsing tests**
- Covers natural language time expressions, relative dates, and edge cases
- Tests date range validation and timezone handling
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.time_parser`
- **Runtime**: Fast

### `test_summarizer.py` - Summarizer Tests
- **3 focused unit tests**
- Tests message summarization functionality
- Covers empty ranges and basic summarization logic
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.summarizer`  
- **Runtime**: Fast

### `test_utils.py` - Utility Tests
- **3 utility function tests**
- Tests helper functions like URL building
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
