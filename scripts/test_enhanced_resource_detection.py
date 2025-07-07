#!/usr/bin/env python3
"""
Enhanced Resource Detection Test Script

Comprehensive testing of the improved resource detection pipeline
with real-world examples and quality validation.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from agentic.services.content_processor import ContentProcessingService
from agentic.services.enhanced_resource_detector import EnhancedResourceDetector, ResourceType
from agentic.services.resource_validation_pipeline import ResourceValidationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedResourceDetectionTester:
    """
    Comprehensive tester for enhanced resource detection system
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.content_processor = ContentProcessingService(self.openai_client)
        self.resource_detector = EnhancedResourceDetector(self.openai_client)
        self.validation_pipeline = ResourceValidationPipeline()
        
        # Test cases representing real Discord messages
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """
        Create comprehensive test cases covering different resource types
        """
        return [
            # High-quality academic resource
            {
                "name": "Academic Research Paper",
                "message": {
                    "message_id": "test_001",
                    "content": "Check out this groundbreaking research on transformer architecture: https://arxiv.org/abs/1706.03762 - the 'Attention Is All You Need' paper that revolutionized NLP. The authors introduce the Transformer model which relies entirely on attention mechanisms.",
                    "attachments": [],
                    "author": {"username": "researcher_ai"},
                    "channel_id": "research-papers",
                    "timestamp": "2025-01-15T10:00:00Z"
                },
                "expected_resources": 1,
                "expected_types": [ResourceType.PREPRINT],
                "expected_min_quality": 0.8
            },
            
            # Code repository with explanation
            {
                "name": "Code Repository Share",
                "message": {
                    "message_id": "test_002",
                    "content": "Here's my implementation of a transformer from scratch: https://github.com/user/transformer-implementation\n\n```python\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.n_heads = n_heads\n        self.d_k = d_model // n_heads\n        \n    def forward(self, query, key, value, mask=None):\n        batch_size = query.size(0)\n        # Implementation continues...\n```\n\nThis includes attention mechanism, positional encoding, and training loop.",
                    "attachments": [],
                    "author": {"username": "ml_engineer"},
                    "channel_id": "code-sharing",
                    "timestamp": "2025-01-15T11:00:00Z"
                },
                "expected_resources": 2,
                "expected_types": [ResourceType.CODE_REPOSITORY, ResourceType.CODE_SNIPPET],
                "expected_min_quality": 0.7
            },
            
            # Dataset and model showcase
            {
                "name": "AI Model and Dataset",
                "message": {
                    "message_id": "test_003",
                    "content": "Trained a new model on ImageNet dataset using PyTorch. Check out the model card: https://huggingface.co/user/vision-transformer-base\n\nAchieved 94.2% accuracy on ImageNet validation set. The model uses Vision Transformer architecture with 12 layers and 768 hidden dimensions.",
                    "attachments": [],
                    "author": {"username": "cv_researcher"},
                    "channel_id": "model-showcase",
                    "timestamp": "2025-01-15T12:00:00Z"
                },
                "expected_resources": 3,  # Model, Dataset mention, Framework mention
                "expected_types": [ResourceType.MODEL, ResourceType.DATASET],
                "expected_min_quality": 0.6
            },
            
            # Tutorial with multiple resources
            {
                "name": "Comprehensive Tutorial",
                "message": {
                    "message_id": "test_004",
                    "content": "Great tutorial series on deep learning fundamentals: https://www.deeplearning.ai/courses/deep-learning-specialization/\n\nAlso check out this YouTube playlist: https://www.youtube.com/watch?v=example123&list=PLplaylist\n\nFor hands-on practice, I recommend this Kaggle competition: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2",
                    "attachments": [
                        {
                            "filename": "deep_learning_notes.pdf",
                            "size": 2048576,
                            "url": "https://cdn.discordapp.com/attachments/123/notes.pdf",
                            "content_type": "application/pdf"
                        }
                    ],
                    "author": {"username": "ai_educator"},
                    "channel_id": "learning-resources",
                    "timestamp": "2025-01-15T13:00:00Z"
                },
                "expected_resources": 4,
                "expected_types": [ResourceType.COURSE, ResourceType.VIDEO_TUTORIAL, ResourceType.DATASET, ResourceType.PDF_DOCUMENT],
                "expected_min_quality": 0.7
            },
            
            # Industry report and whitepaper
            {
                "name": "Industry Report",
                "message": {
                    "message_id": "test_005",
                    "content": "New McKinsey report on AI adoption in enterprise: https://www.mckinsey.com/capabilities/quantumblack/our-insights/ai-adoption-report-2024\n\nKey findings:\n- 65% of organizations are now using AI in at least one business function\n- Generative AI adoption has doubled in the last year\n- ROI on AI investments averaging 20% annually",
                    "attachments": [
                        {
                            "filename": "AI_Enterprise_Adoption_Summary.pptx",
                            "size": 5242880,
                            "url": "https://cdn.discordapp.com/attachments/456/presentation.pptx",
                            "content_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        }
                    ],
                    "author": {"username": "business_analyst"},
                    "channel_id": "industry-insights",
                    "timestamp": "2025-01-15T14:00:00Z"
                },
                "expected_resources": 2,
                "expected_types": [ResourceType.INDUSTRY_REPORT, ResourceType.PRESENTATION],
                "expected_min_quality": 0.6
            },
            
            # Low-quality/spam content (should be filtered)
            {
                "name": "Low Quality Content",
                "message": {
                    "message_id": "test_006",
                    "content": "lol check this funny meme https://tenor.com/view/funny-cat-gif also click here to win free money https://spam.com/free-money",
                    "attachments": [],
                    "author": {"username": "spam_user"},
                    "channel_id": "general-chat",
                    "timestamp": "2025-01-15T15:00:00Z"
                },
                "expected_resources": 0,  # Should be filtered out
                "expected_types": [],
                "expected_min_quality": 0.0
            },
            
            # Mixed quality content
            {
                "name": "Mixed Quality Content",
                "message": {
                    "message_id": "test_007",
                    "content": "Working on a new project. Here's the repo: https://github.com/user/awesome-project\n\nAlso found this random link: https://example.com/random-page\n\nAnd here's a useful Stack Overflow discussion: https://stackoverflow.com/questions/12345/transformer-implementation",
                    "attachments": [],
                    "author": {"username": "developer"},
                    "channel_id": "project-updates",
                    "timestamp": "2025-01-15T16:00:00Z"
                },
                "expected_resources": 2,  # GitHub and Stack Overflow should be kept
                "expected_types": [ResourceType.CODE_REPOSITORY, ResourceType.FORUM_POST],
                "expected_min_quality": 0.4
            }
        ]
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite
        """
        print("🧪 Starting Enhanced Resource Detection Test Suite")
        print("=" * 60)
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {
                "total_resources_detected": 0,
                "average_quality_score": 0.0,
                "average_processing_time": 0.0
            }
        }
        
        total_processing_time = 0.0
        total_quality_scores = []
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\n📝 Test {i+1}/{len(self.test_cases)}: {test_case['name']}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Run the test
            result = await self._run_single_test(test_case)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            total_processing_time += processing_time
            
            # Collect quality scores
            if result["quality_score"] > 0:
                total_quality_scores.append(result["quality_score"])
            
            test_results["test_details"].append(result)
            test_results["performance_metrics"]["total_resources_detected"] += result["resources_detected"]
            
            if result["passed"]:
                test_results["passed_tests"] += 1
                print(f"✅ PASSED - {result['resources_detected']} resources detected")
            else:
                test_results["failed_tests"] += 1
                print(f"❌ FAILED - {result['failure_reason']}")
        
        # Calculate performance metrics
        test_results["performance_metrics"]["average_processing_time"] = total_processing_time / len(self.test_cases)
        test_results["performance_metrics"]["average_quality_score"] = (
            sum(total_quality_scores) / len(total_quality_scores) if total_quality_scores else 0.0
        )
        
        # Generate summary
        self._print_test_summary(test_results)
        
        return test_results
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case
        """
        message = test_case["message"]
        expected_resources = test_case["expected_resources"]
        expected_types = test_case["expected_types"]
        expected_min_quality = test_case["expected_min_quality"]
        
        try:
            # Analyze message content
            analysis = await self.content_processor.analyze_message_content(message)
            
            # Extract resource information
            resources_detected = len(analysis.get("resource_metadata", []))
            detected_types = []
            quality_scores = []
            
            for metadata in analysis.get("resource_metadata", []):
                resource = metadata["resource"]
                detected_types.append(resource.type)
                quality_scores.append(resource.quality_score)
            
            overall_quality = analysis.get("quality_score", 0.0)
            
            # Validate test expectations
            passed = True
            failure_reasons = []
            
            # Check resource count
            if resources_detected != expected_resources:
                passed = False
                failure_reasons.append(f"Expected {expected_resources} resources, got {resources_detected}")
            
            # Check resource types
            for expected_type in expected_types:
                if expected_type not in detected_types:
                    passed = False
                    failure_reasons.append(f"Expected resource type {expected_type.value} not found")
            
            # Check minimum quality
            if overall_quality < expected_min_quality:
                passed = False
                failure_reasons.append(f"Quality score {overall_quality:.2f} below expected minimum {expected_min_quality}")
            
            return {
                "test_name": test_case["name"],
                "passed": passed,
                "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
                "resources_detected": resources_detected,
                "detected_types": [t.value for t in detected_types],
                "quality_score": overall_quality,
                "individual_quality_scores": quality_scores,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Test error for {test_case['name']}: {e}")
            return {
                "test_name": test_case["name"],
                "passed": False,
                "failure_reason": f"Exception: {str(e)}",
                "resources_detected": 0,
                "detected_types": [],
                "quality_score": 0.0,
                "individual_quality_scores": [],
                "analysis": {}
            }
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """
        Print comprehensive test summary
        """
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {(results['passed_tests'] / results['total_tests']) * 100:.1f}%")
        
        print(f"\n📈 PERFORMANCE METRICS")
        print(f"Total Resources Detected: {results['performance_metrics']['total_resources_detected']}")
        print(f"Average Quality Score: {results['performance_metrics']['average_quality_score']:.3f}")
        print(f"Average Processing Time: {results['performance_metrics']['average_processing_time']:.3f}s")
        
        print(f"\n🎯 DETAILED RESULTS")
        for detail in results['test_details']:
            status = "✅ PASS" if detail['passed'] else "❌ FAIL"
            print(f"{status} - {detail['test_name']}: {detail['resources_detected']} resources, quality {detail['quality_score']:.2f}")
            if not detail['passed']:
                print(f"   Reason: {detail['failure_reason']}")
        
        print(f"\n💡 RECOMMENDATIONS")
        self._generate_recommendations(results)
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """
        Generate improvement recommendations based on test results
        """
        success_rate = results['passed_tests'] / results['total_tests']
        avg_quality = results['performance_metrics']['average_quality_score']
        avg_time = results['performance_metrics']['average_processing_time']
        
        recommendations = []
        
        if success_rate < 0.8:
            recommendations.append("⚠️  Test success rate is below 80% - review detection algorithms")
        
        if avg_quality < 0.6:
            recommendations.append("📊 Average quality score is low - improve quality scoring methodology")
        
        if avg_time > 2.0:
            recommendations.append("⏱️  Processing time is high - consider performance optimization")
        
        # Check for specific failure patterns
        failed_tests = [d for d in results['test_details'] if not d['passed']]
        if failed_tests:
            failure_types = {}
            for test in failed_tests:
                reason = test['failure_reason']
                if reason:
                    failure_types[reason] = failure_types.get(reason, 0) + 1
            
            for reason, count in failure_types.items():
                recommendations.append(f"🔍 {count} tests failed due to: {reason}")
        
        if not recommendations:
            recommendations.append("🎉 All tests performing well - system is operating optimally")
        
        for rec in recommendations:
            print(f"   {rec}")
    
    async def benchmark_performance(self) -> Dict[str, Any]:
        """
        Benchmark performance against current system
        """
        print("\n🏁 PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Test with a sample of real messages
        sample_messages = [
            {
                "message_id": "benchmark_001",
                "content": "Check out this research paper: https://arxiv.org/abs/2024.12345 and this code: https://github.com/user/repo with some inline code `print('hello')`",
                "attachments": [],
                "author": {"username": "test_user"},
                "channel_id": "test_channel",
                "timestamp": "2025-01-15T10:00:00Z"
            }
        ]
        
        # Benchmark processing time
        iterations = 10
        total_time = 0.0
        
        for i in range(iterations):
            start_time = asyncio.get_event_loop().time()
            
            for message in sample_messages:
                await self.content_processor.analyze_message_content(message)
            
            iteration_time = asyncio.get_event_loop().time() - start_time
            total_time += iteration_time
        
        avg_time_per_message = total_time / (iterations * len(sample_messages))
        
        print(f"Average processing time per message: {avg_time_per_message:.3f}s")
        print(f"Estimated throughput: {1/avg_time_per_message:.1f} messages/second")
        
        return {
            "average_processing_time": avg_time_per_message,
            "estimated_throughput": 1/avg_time_per_message,
            "total_benchmark_time": total_time
        }
    
    async def save_test_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save test results to file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_resource_detection_test_{timestamp}.json"
        
        # Convert non-serializable objects
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Test results saved to: {filename}")
    
    async def cleanup(self):
        """
        Cleanup resources
        """
        await self.content_processor.cleanup()
        await self.resource_detector.cleanup()
        await self.validation_pipeline.cleanup()


async def main():
    """
    Main test execution
    """
    tester = EnhancedResourceDetectionTester()
    
    try:
        # Run comprehensive test suite
        test_results = await tester.run_comprehensive_test()
        
        # Run performance benchmark
        benchmark_results = await tester.benchmark_performance()
        
        # Combine results
        full_results = {
            "test_results": test_results,
            "benchmark_results": benchmark_results
        }
        
        # Save results
        await tester.save_test_results(full_results)
        
        print("\n🎉 Enhanced Resource Detection Testing Complete!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 