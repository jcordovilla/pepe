"""
Main Performance Test Orchestrator

Coordinates the execution of the complete performance test suite.
Orchestrates content analysis, query generation, bot execution, evaluation, and reporting.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import test suite components
from tests.performance_test_suite.content_analyzer import ContentAnalyzer
from tests.performance_test_suite.query_generator import QueryGenerator
from tests.performance_test_suite.bot_runner import BotRunner
from tests.performance_test_suite.evaluator import ResponseEvaluator
from tests.performance_test_suite.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class PerformanceTestOrchestrator:
    """
    Main orchestrator for the comprehensive performance test suite.
    
    Coordinates:
    1. Content analysis of Discord server
    2. Generation of 50 diverse test queries
    3. Execution of queries against the bot
    4. Evaluation of responses against expected answers
    5. Generation of comprehensive report with recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        
        # Initialize components
        self.content_analyzer = ContentAnalyzer(self.config.get("database_path"))
        self.content_analyzer.preflight_schema_check()
        self.query_generator = QueryGenerator()
        self.bot_runner = BotRunner(self.config.get("bot_api_endpoint"))
        self.evaluator = ResponseEvaluator()
        self.report_generator = ReportGenerator()
        
        logger.info("PerformanceTestOrchestrator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the test suite."""
        return {
            "database_path": "data/discord_messages.db",
            "bot_api_endpoint": "http://localhost:8000",
            "output_directory": "tests/performance_test_suite/data",
            "sample_percentage": 0.15,
            "query_count": 50,
            "enable_error_scenarios": True,
            "save_intermediate_results": True
        }
    
    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete performance test suite.
        
        Returns:
            Dictionary containing all test results and report
        """
        logger.info("Starting complete performance test suite execution...")
        
        try:
            # Phase 1: Content Analysis
            logger.info("=== PHASE 1: Content Analysis ===")
            content_analysis = await self._run_content_analysis()
            
            # Phase 2: Query Generation
            logger.info("=== PHASE 2: Query Generation ===")
            queries = await self._run_query_generation(content_analysis)
            
            # Phase 3: Bot Execution
            logger.info("=== PHASE 3: Bot Execution ===")
            bot_responses = await self._run_bot_execution(queries)
            
            # Phase 4: Response Evaluation
            logger.info("=== PHASE 4: Response Evaluation ===")
            evaluation_results = await self._run_response_evaluation(queries, bot_responses)
            
            # Phase 5: Report Generation
            logger.info("=== PHASE 5: Report Generation ===")
            comprehensive_report = await self._run_report_generation(
                content_analysis, queries, bot_responses, evaluation_results
            )
            
            # Compile final results
            self.results = {
                "test_execution": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration": "Complete test suite execution",
                    "phases_completed": 5
                },
                "content_analysis": content_analysis,
                "queries": queries,
                "bot_responses": bot_responses,
                "evaluation_results": evaluation_results,
                "comprehensive_report": comprehensive_report,
                "summary": {
                    "total_queries": len(queries),
                    "successful_responses": len([r for r in bot_responses if r.success]),
                    "average_quality_score": comprehensive_report.evaluation_summary.get("overall_performance", {}).get("average_score", 0),
                    "system_reliability": comprehensive_report.test_summary["overall_performance"]["system_reliability"]
                }
            }
            
            # Save all results
            await self._save_all_results()
            
            logger.info("Complete performance test suite execution finished successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in test suite execution: {e}")
            raise
    
    async def _run_content_analysis(self) -> Dict[str, Any]:
        """Run content analysis phase."""
        logger.info("Starting content analysis of Discord server...")
        
        try:
            content_analysis = await self.content_analyzer.analyze_server_content()
            
            # Save content analysis results
            if self.config.get("save_intermediate_results"):
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config['output_directory']}/content_analysis_{timestamp}.json"
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                
                with open(filename, 'w') as f:
                    json.dump(content_analysis, f, indent=2, default=str)
                
                logger.info(f"Content analysis results saved to: {filename}")
            
            logger.info("Content analysis completed successfully")
            return content_analysis
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            raise
    
    async def _run_query_generation(self, content_analysis: Dict[str, Any]) -> List[Any]:
        """Run query generation phase."""
        logger.info("Starting query generation...")
        
        try:
            # Update query generator with content analysis
            self.query_generator.content_analysis = content_analysis
            
            # Generate test queries
            queries = self.query_generator.generate_test_queries()
            
            # Save queries
            if self.config.get("save_intermediate_results"):
                queries_filename = self.query_generator.save_queries()
                logger.info(f"Test queries saved to: {queries_filename}")
            
            logger.info(f"Generated {len(queries)} test queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error in query generation: {e}")
            raise
    
    async def _run_bot_execution(self, queries: List[Any]) -> List[Any]:
        """Run bot execution phase."""
        logger.info("Starting bot execution...")
        
        try:
            # Execute main queries
            bot_responses = await self.bot_runner.run_queries(queries)
            
            # Run error scenarios if enabled
            if self.config.get("enable_error_scenarios"):
                logger.info("Running error scenario tests...")
                error_responses = await self.bot_runner.run_error_scenarios()
                bot_responses.extend(error_responses)
            
            # Save bot responses
            if self.config.get("save_intermediate_results"):
                responses_filename = self.bot_runner.save_responses()
                logger.info(f"Bot responses saved to: {responses_filename}")
            
            logger.info(f"Bot execution completed: {len(bot_responses)} responses collected")
            return bot_responses
            
        except Exception as e:
            logger.error(f"Error in bot execution: {e}")
            raise
    
    async def _run_response_evaluation(self, queries: List[Any], bot_responses: List[Any]) -> List[Any]:
        """Run response evaluation phase."""
        logger.info("Starting response evaluation...")
        
        try:
            # Filter responses to only include main queries (not error scenarios)
            main_responses = [r for r in bot_responses if isinstance(r.query_id, int)]
            
            # Evaluate responses
            evaluation_results = self.evaluator.evaluate_responses(queries, main_responses)
            
            # Save evaluation results
            if self.config.get("save_intermediate_results"):
                evaluation_filename = self.evaluator.save_evaluation_results()
                logger.info(f"Evaluation results saved to: {evaluation_filename}")
            
            logger.info(f"Response evaluation completed: {len(evaluation_results)} evaluations")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in response evaluation: {e}")
            raise
    
    async def _run_report_generation(
        self,
        content_analysis: Dict[str, Any],
        queries: List[Any],
        bot_responses: List[Any],
        evaluation_results: List[Any]
    ) -> Any:
        """Run report generation phase."""
        logger.info("Starting report generation...")
        
        try:
            # Get summaries
            content_summary = content_analysis.get("server_overview", {})
            query_summary = self.query_generator.get_query_summary()
            execution_summary = self.bot_runner.get_execution_summary()
            evaluation_summary = self.evaluator.get_evaluation_summary()
            
            # Generate comprehensive report
            comprehensive_report = self.report_generator.generate_comprehensive_report(
                content_analysis=content_summary,
                query_summary=query_summary,
                execution_summary=execution_summary,
                evaluation_summary=evaluation_summary,
                evaluation_results=evaluation_results
            )
            
            # Save comprehensive report
            if self.config.get("save_intermediate_results"):
                report_filename = self.report_generator.save_report()
                logger.info(f"Comprehensive report saved to: {report_filename}")
            
            # Generate and save executive summary
            executive_summary = self.report_generator.generate_executive_summary()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            summary_filename = f"{self.config['output_directory']}/executive_summary_{timestamp}.md"
            Path(summary_filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_filename, 'w') as f:
                f.write(executive_summary)
            
            logger.info(f"Executive summary saved to: {summary_filename}")
            logger.info("Report generation completed successfully")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            raise
    
    async def _save_all_results(self):
        """Save all test results to a single file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['output_directory']}/complete_test_results_{timestamp}.json"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Complete test results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving complete results: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.bot_runner.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of the test execution."""
        if not self.results:
            return {"status": "No test results available"}
        
        return {
            "status": "Test completed successfully",
            "timestamp": self.results["test_execution"]["timestamp"],
            "summary": self.results["summary"],
            "key_findings": self.results["comprehensive_report"].key_findings[:5],
            "top_recommendations": [
                rec.title for rec in self.results["comprehensive_report"].recommendations[:3]
            ]
        }


async def main():
    """Main entry point for the performance test suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tests/performance_test_suite/performance_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Starting Performance Test Suite")
    
    # Configuration
    config = {
        "database_path": "data/discord_messages.db",
        "bot_api_endpoint": "http://localhost:8000",
        "output_directory": "tests/performance_test_suite/data",
        "sample_percentage": 0.15,
        "query_count": 50,
        "enable_error_scenarios": True,
        "save_intermediate_results": True
    }
    
    # Create orchestrator
    orchestrator = PerformanceTestOrchestrator(config)
    
    try:
        # Run complete test suite
        results = await orchestrator.run_complete_test_suite()
        
        # Print summary
        summary = orchestrator.get_test_summary()
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUITE - EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {summary['status']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Queries: {summary['summary']['total_queries']}")
        print(f"Successful Responses: {summary['summary']['successful_responses']}")
        print(f"Average Quality Score: {summary['summary']['average_quality_score']:.2f}")
        print(f"System Reliability: {summary['summary']['system_reliability']:.2f}")
        
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  - {finding}")
        
        print("\nTop Recommendations:")
        for rec in summary['top_recommendations']:
            print(f"  - {rec}")
        
        print("\n" + "="*60)
        print("Test suite execution completed successfully!")
        print("Check the generated reports for detailed analysis and recommendations.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"\nERROR: Test suite execution failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 