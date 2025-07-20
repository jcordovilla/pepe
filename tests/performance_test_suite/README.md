# Performance Test Suite

A comprehensive testing framework for evaluating the agentic Discord bot performance. This test suite systematically analyzes server content, generates diverse test queries, executes them against the bot, and provides detailed evaluation with actionable recommendations.

## 🎯 Objectives

The test suite addresses the following objectives:

1. **Systematic Content Analysis**: Analyzes Discord server structure and content patterns
2. **Comprehensive Query Generation**: Creates 50 diverse test queries covering all bot functionalities
3. **Bot Performance Testing**: Executes queries and collects responses with error handling
4. **Response Evaluation**: Evaluates actual responses against expected dummy answers
5. **Actionable Recommendations**: Provides detailed conclusions and improvement strategies

## 📋 Test Coverage

The test suite covers all bot functionalities:

- 📊 **Server Data Analysis**: Monitor and gather insights from messages, reactions, threads
- 📝 **Feedback Summarization**: Create summaries of feedback and posts
- 📈 **Trending Topics**: Recognize and track trending topics within discussions
- ❓ **Q&A Concepts**: Collect questions and answers related to AI
- 🔢 **Statistics**: Generate statistics on server activity and engagement
- 🌐 **Server Structure Analysis**: Examine server structure and dynamics
- 📰 **Digests**: Provide digests of server, channel, or user activity
- 🗣️ **Conversational Leadership**: Assist in distilling high-value topics

## 🏗️ Architecture

The test suite consists of five main components:

```
performance_test_suite/
├── content_analyzer.py      # Analyzes Discord server content
├── query_generator.py       # Generates 50 diverse test queries
├── bot_runner.py           # Executes queries against the bot
├── evaluator.py            # Evaluates responses against expectations
├── report_generator.py     # Generates comprehensive reports
├── main_orchestrator.py    # Coordinates all components
├── run_tests.py           # Simple runner script
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Discord Messages Database**: Ensure `data/discord_messages.db` exists
2. **Bot API Running**: Start the Discord bot API on `http://localhost:8000`
3. **Python Dependencies**: Install required packages

### Basic Usage

```bash
# Run with default configuration
python tests/performance_test_suite/run_tests.py

# Run with custom configuration
python tests/performance_test_suite/run_tests.py --config config.json

# Run with specific parameters
python tests/performance_test_suite/run_tests.py \
  --database data/discord_messages.db \
  --bot-endpoint http://localhost:8000 \
  --output-dir tests/performance_test_suite/data
```

### Configuration

Create a `config.json` file to customize the test suite:

```json
{
  "database_path": "data/discord_messages.db",
  "bot_api_endpoint": "http://localhost:8000",
  "output_directory": "tests/performance_test_suite/data",
  "sample_percentage": 0.15,
  "query_count": 50,
  "enable_error_scenarios": true,
  "save_intermediate_results": true
}
```

## 📊 Test Phases

### Phase 1: Content Analysis
- Analyzes 15% of total messages (minimum 1000)
- Examines server structure, channels, and user activity
- Identifies content patterns and topic distribution
- Generates insights for query generation

### Phase 2: Query Generation
- Creates 50 balanced test queries across all functionalities
- Includes edge cases and real user behavior patterns
- Generates expected response structures for each query
- Covers simple, moderate, and complex query types

### Phase 3: Bot Execution
- Executes queries sequentially against the bot API
- Collects responses with timing and metadata
- Handles errors and edge scenarios
- Measures response times and success rates

### Phase 4: Response Evaluation
- Evaluates responses against expected structures
- Calculates multiple quality metrics:
  - Relevance (25% weight)
  - Format accuracy (20% weight)
  - Completeness (20% weight)
  - Coherence (15% weight)
  - Semantic similarity (15% weight)
  - Performance (5% weight)

### Phase 5: Report Generation
- Generates comprehensive performance report
- Provides actionable recommendations
- Identifies architecture insights
- Creates executive summary

## 📈 Evaluation Metrics

### Response Quality Metrics

1. **Relevance Score**: How well the response addresses the query
2. **Format Score**: Accuracy of response structure and formatting
3. **Completeness Score**: Coverage of expected information
4. **Coherence Score**: Readability and logical flow
5. **Semantic Similarity**: Content matching vs. semantic relevance
6. **Performance Score**: Response time and efficiency

### System Reliability Metrics

- **Success Rate**: Percentage of successful responses
- **Error Classification**: Types and frequency of errors
- **Response Time**: Average, min, max response times
- **Error Recovery**: System's ability to handle failures

## 📁 Output Files

The test suite generates several output files:

```
tests/performance_test_suite/data/
├── content_analysis_YYYYMMDD_HHMMSS.json
├── test_queries_YYYYMMDD_HHMMSS.json
├── bot_responses_YYYYMMDD_HHMMSS.json
├── evaluation_results_YYYYMMDD_HHMMSS.json
├── comprehensive_report_YYYYMMDD_HHMMSS.json
├── executive_summary_YYYYMMDD_HHMMSS.md
├── complete_test_results_YYYYMMDD_HHMMSS.json
└── performance_test.log
```

## 🔍 Sample Queries

The test suite generates queries like:

- "What are the most active channels in the server?"
- "Show me user engagement patterns over the last month"
- "Create a weekly digest of server activity"
- "Identify high-value topics from recent discussions"
- "What questions remain unanswered?"
- "Analyze the server's channel structure and organization"

## 📊 Sample Report Structure

### Executive Summary
- Overall performance metrics
- Key findings and insights
- Top recommendations
- Next steps

### Detailed Analysis
- Category-wise performance breakdown
- Complexity analysis
- Error pattern analysis
- Quality distribution

### Recommendations
- High-priority improvements
- Medium-priority enhancements
- Low-effort optimizations
- Architectural improvements

## 🛠️ Customization

### Adding New Query Types

1. Extend `QueryGenerator` class
2. Add new query generation methods
3. Update expected response structures
4. Add evaluation criteria

### Modifying Evaluation Criteria

1. Update `ResponseEvaluator` class
2. Adjust metric weights
3. Add new evaluation methods
4. Update scoring algorithms

### Custom Configuration

1. Create custom config file
2. Modify default parameters
3. Add new configuration options
4. Update orchestrator logic

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure `data/discord_messages.db` exists
   - Check database permissions
   - Verify SQLite installation

2. **Bot API Connection Error**
   - Ensure bot API is running
   - Check endpoint URL
   - Verify network connectivity

3. **Import Errors**
   - Check Python path
   - Install missing dependencies
   - Verify file structure

### Debug Mode

Enable debug logging by modifying the logging configuration in `main_orchestrator.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Logging

The test suite provides comprehensive logging:

- **File Logging**: `tests/performance_test_suite/performance_test.log`
- **Console Output**: Real-time progress and results
- **Error Tracking**: Detailed error information
- **Performance Metrics**: Timing and statistics

## 🤝 Contributing

To extend the test suite:

1. **Add New Components**: Create new analysis or evaluation modules
2. **Enhance Metrics**: Improve evaluation algorithms
3. **Add Test Cases**: Include new query types or scenarios
4. **Improve Reporting**: Enhance report generation and visualization

## 📄 License

This test suite is part of the Discord Bot Agentic project and follows the same licensing terms.

## 🆘 Support

For issues or questions:

1. Check the troubleshooting section
2. Review the generated logs
3. Examine the detailed reports
4. Consult the main project documentation

---

**Note**: This test suite is designed to provide comprehensive insights into bot performance and should be run regularly to monitor system health and identify improvement opportunities. 