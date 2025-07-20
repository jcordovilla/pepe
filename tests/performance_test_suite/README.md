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
4. **Llama Model** (Optional): Install Ollama and Llama model for AI-powered evaluation
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama model
   ollama pull llama3.2:3b
   ```

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
- Evaluates responses against expected structures using hybrid approach
- **Llama AI Evaluation** for semantic understanding:
  - Semantic relevance (30% weight)
  - Response quality (25% weight)
  - Coherence evaluation (10% weight)
- **Rule-based Evaluation** for objective metrics:
  - Format accuracy (15% weight)
  - Completeness (15% weight)
  - Performance (5% weight)
- Fallback to rule-based evaluation if Llama unavailable

### Phase 5: Report Generation
- Generates comprehensive performance report
- Provides actionable recommendations
- Identifies architecture insights
- Creates executive summary

## 📈 Evaluation Metrics

### Response Quality Metrics

1. **Relevance Score** (30% weight): Semantic relevance using Llama model
2. **Quality Score** (25% weight): Response quality assessment using Llama model
3. **Format Score** (15% weight): Accuracy of response structure and formatting
4. **Completeness Score** (15% weight): Coverage of expected information
5. **Coherence Score** (10% weight): Logical flow using Llama model
6. **Performance Score** (5% weight): Response time and efficiency

### AI-Powered Evaluation

The test suite uses **Llama model integration** for semantic evaluation:

- **Semantic Relevance**: Llama evaluates how well responses address queries
- **Quality Assessment**: Llama rates response accuracy, helpfulness, and appropriateness
- **Coherence Evaluation**: Llama assesses logical flow and organization
- **Fallback Mode**: Rule-based evaluation when Llama is unavailable

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

## 🤖 Llama Integration

### AI-Powered Evaluation Benefits

- **Semantic Understanding**: True comprehension of query-response relationships
- **Quality Assessment**: Evaluates accuracy, helpfulness, and appropriateness
- **Context Awareness**: Understands Discord-specific context and tone
- **Nuanced Scoring**: Distinguishes between good but incomplete vs. wrong but well-formatted responses
- **Fallback Support**: Graceful degradation to rule-based evaluation when AI unavailable

### Llama Model Configuration

```python
# Default model
llama_evaluator = LlamaEvaluator(model_name="llama3.2:3b")

# Custom model
llama_evaluator = LlamaEvaluator(model_name="llama3.1:8b")
```

### Evaluation Prompts

The Llama model uses carefully crafted prompts for:
- **Semantic Relevance**: "Rate how well this response answers the query"
- **Quality Assessment**: "Rate the quality of this Discord bot response"
- **Coherence Evaluation**: "Rate the coherence and logical flow"

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
5. Customize Llama evaluation prompts

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