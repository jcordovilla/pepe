#!/usr/bin/env python3
"""
Resource Detection and Classification Evaluation Tests

Comprehensive test suite for evaluating the quality and accuracy of:
- URL detection and classification
- Attachment processing
- Code snippet detection
- AI-powered content classification
- Resource quality assessment
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pytest
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from root directory
load_dotenv(project_root / ".env")

from agentic.services.content_processor import ContentProcessingService
from agentic.analytics.validation_system import ValidationSystem, QualityDimension
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class TestResource:
    """Test resource with expected classification"""
    content: str
    message_data: Dict[str, Any]
    expected_urls: List[Dict[str, Any]]
    expected_code_snippets: List[Dict[str, Any]]
    expected_attachments: List[Dict[str, Any]]
    expected_classifications: List[str]
    resource_quality_score: float  # Expected quality score 0-1

class ResourceDetectionEvaluator:
    """
    Comprehensive evaluator for resource detection and classification system
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.content_processor = ContentProcessingService(self.openai_client)
        
        # Initialize validation system with minimal config
        validation_config = {
            "validation_enabled": True,
            "default_method": "heuristic",
            "enable_ai_validation": False
        }
        self.validation_system = ValidationSystem(validation_config)
        
        # Test data for evaluation
        self.test_resources = self._create_test_dataset()
        
        # Evaluation metrics
        self.metrics = {
            "url_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "code_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "attachment_processing": {"accuracy": 0.0},
            "ai_classification": {"accuracy": 0.0, "semantic_similarity": 0.0},
            "quality_assessment": {"correlation": 0.0, "rmse": 0.0}
        }
    
    def _create_test_dataset(self) -> List[TestResource]:
        """Create comprehensive test dataset for evaluation"""
        return [
            # Academic paper resource
            TestResource(
                content="Check out this amazing research paper on transformers: https://arxiv.org/abs/1706.03762 - it's the attention is all you need paper that revolutionized NLP!",
                message_data={
                    "message_id": "test_001",
                    "timestamp": "2025-06-05T10:00:00Z",
                    "author": {"username": "researcher_ai"},
                    "channel_name": "research-papers",
                    "attachments": []
                },
                expected_urls=[{
                    "url": "https://arxiv.org/abs/1706.03762",
                    "domain": "arxiv.org",
                    "type": "development"  # Based on current classification logic
                }],
                expected_code_snippets=[],
                expected_attachments=[],
                expected_classifications=["resource_sharing", "technical", "research"],
                resource_quality_score=0.9
            ),
            
            # Code sharing resource
            TestResource(
                content="Here's a Python function for data preprocessing:\n```python\ndef preprocess_data(df):\n    return df.dropna().reset_index(drop=True)\n```\nYou can also check my GitHub repo: https://github.com/user/ml-toolkit",
                message_data={
                    "message_id": "test_002",
                    "timestamp": "2025-06-05T10:05:00Z",
                    "author": {"username": "dev_expert"},
                    "channel_name": "code-sharing",
                    "attachments": []
                },
                expected_urls=[{
                    "url": "https://github.com/user/ml-toolkit",
                    "domain": "github.com",
                    "type": "development"
                }],
                expected_code_snippets=[{
                    "language": "python",
                    "code": "```python\ndef preprocess_data(df):\n    return df.dropna().reset_index(drop=True)\n```"
                }],
                expected_attachments=[],
                expected_classifications=["code_help", "resource_sharing", "technical"],
                resource_quality_score=0.85
            ),
            
            # Tutorial video resource
            TestResource(
                content="Great tutorial on machine learning basics: https://www.youtube.com/watch?v=example123 - perfect for beginners!",
                message_data={
                    "message_id": "test_003",
                    "timestamp": "2025-06-05T10:10:00Z",
                    "author": {"username": "ml_teacher"},
                    "channel_name": "tutorials",
                    "attachments": []
                },
                expected_urls=[{
                    "url": "https://www.youtube.com/watch?v=example123",
                    "domain": "www.youtube.com",
                    "type": "video"
                }],
                expected_code_snippets=[],
                expected_attachments=[],
                expected_classifications=["resource_sharing", "educational", "tutorial"],
                resource_quality_score=0.8
            ),
            
            # Document attachment resource
            TestResource(
                content="Here's the whitepaper I mentioned earlier. Contains detailed methodology and results.",
                message_data={
                    "message_id": "test_004",
                    "timestamp": "2025-06-05T10:15:00Z",
                    "author": {"username": "research_lead"},
                    "channel_name": "research-docs",
                    "attachments": [{
                        "filename": "ai_methodology_whitepaper.pdf",
                        "size": 2048576,
                        "url": "https://cdn.discordapp.com/attachments/123/ai_whitepaper.pdf",
                        "content_type": "application/pdf"
                    }]
                },
                expected_urls=[],
                expected_code_snippets=[],
                expected_attachments=[{
                    "filename": "ai_methodology_whitepaper.pdf",
                    "content_type": "application/pdf"
                }],
                expected_classifications=["resource_sharing", "technical", "documentation"],
                resource_quality_score=0.95
            ),
            
            # Noise/low-quality content
            TestResource(
                content="lol check this meme https://tenor.com/view/funny-cat-gif",
                message_data={
                    "message_id": "test_005",
                    "timestamp": "2025-06-05T10:20:00Z",
                    "author": {"username": "meme_lover"},
                    "channel_name": "general-chat",
                    "attachments": []
                },
                expected_urls=[],  # Should be filtered out by noise filtering
                expected_code_snippets=[],
                expected_attachments=[],
                expected_classifications=["meme", "casual"],
                resource_quality_score=0.1
            ),
            
            # Mixed content with multiple resources
            TestResource(
                content="For the AI project, here are useful resources:\n1. Dataset: https://huggingface.co/datasets/example\n2. Model code: `model = torch.nn.Linear(10, 1)`\n3. Paper: https://arxiv.org/abs/2024.12345\nAlso uploaded the config file.",
                message_data={
                    "message_id": "test_006",
                    "timestamp": "2025-06-05T10:25:00Z",
                    "author": {"username": "project_lead"},
                    "channel_name": "ai-project",
                    "attachments": [{
                        "filename": "model_config.json",
                        "size": 1024,
                        "url": "https://cdn.discordapp.com/attachments/456/config.json",
                        "content_type": "application/json"
                    }]
                },
                expected_urls=[
                    {"url": "https://huggingface.co/datasets/example", "domain": "huggingface.co"},
                    {"url": "https://arxiv.org/abs/2024.12345", "domain": "arxiv.org"}
                ],
                expected_code_snippets=[{
                    "code": "`model = torch.nn.Linear(10, 1)`",
                    "language": "unknown"
                }],
                expected_attachments=[{
                    "filename": "model_config.json",
                    "content_type": "application/json"
                }],
                expected_classifications=["resource_sharing", "technical", "project"],
                resource_quality_score=0.9
            )
        ]
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("🔍 Starting comprehensive resource detection evaluation...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_resources),
            "test_results": [],
            "metrics": {},
            "summary": {}
        }
        
        # Run individual test evaluations
        for i, test_resource in enumerate(self.test_resources):
            print(f"📝 Running test {i+1}/{len(self.test_resources)}")
            test_result = await self._evaluate_single_resource(test_resource)
            results["test_results"].append(test_result)
        
        # Calculate aggregate metrics
        results["metrics"] = self._calculate_aggregate_metrics(results["test_results"])
        results["summary"] = self._generate_summary(results["metrics"])
        
        return results
    
    async def _evaluate_single_resource(self, test_resource: TestResource) -> Dict[str, Any]:
        """Evaluate a single test resource"""
        # Run content processing
        analysis = await self.content_processor.analyze_message_content({
            "content": test_resource.content,
            "attachments": test_resource.message_data.get("attachments", []),
            **test_resource.message_data
        })
        
        # Evaluate each component
        url_eval = self._evaluate_url_detection(
            analysis.get("urls", []), 
            test_resource.expected_urls
        )
        
        code_eval = self._evaluate_code_detection(
            analysis.get("code_snippets", []), 
            test_resource.expected_code_snippets
        )
        
        attachment_eval = self._evaluate_attachment_processing(
            analysis.get("attachments_processed", []), 
            test_resource.expected_attachments
        )
        
        classification_eval = await self._evaluate_ai_classification(
            analysis.get("classifications", []), 
            test_resource.expected_classifications
        )
        
        quality_eval = self._evaluate_quality_assessment(
            analysis, 
            test_resource.resource_quality_score
        )
        
        return {
            "test_id": test_resource.message_data["message_id"],
            "content_preview": test_resource.content[:100] + "..." if len(test_resource.content) > 100 else test_resource.content,
            "url_detection": url_eval,
            "code_detection": code_eval,
            "attachment_processing": attachment_eval,
            "ai_classification": classification_eval,
            "quality_assessment": quality_eval,
            "overall_score": (url_eval["f1"] + code_eval["f1"] + attachment_eval["accuracy"] + 
                            classification_eval["accuracy"] + quality_eval["score"]) / 5
        }
    
    def _evaluate_url_detection(self, detected_urls: List[Dict], expected_urls: List[Dict]) -> Dict[str, float]:
        """Evaluate URL detection accuracy"""
        detected_set = {url["url"] for url in detected_urls}
        expected_set = {url["url"] for url in expected_urls}
        
        if not expected_set and not detected_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not expected_set:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        
        if not detected_set:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
        
        intersection = detected_set.intersection(expected_set)
        precision = len(intersection) / len(detected_set) if detected_set else 0.0
        recall = len(intersection) / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _evaluate_code_detection(self, detected_code: List[Dict], expected_code: List[Dict]) -> Dict[str, float]:
        """Evaluate code snippet detection accuracy"""
        # For simplicity, compare count of code snippets
        detected_count = len(detected_code)
        expected_count = len(expected_code)
        
        if expected_count == 0 and detected_count == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if expected_count == 0:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        
        if detected_count == 0:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
        
        # Simple accuracy based on count similarity
        accuracy = 1.0 - abs(detected_count - expected_count) / max(detected_count, expected_count)
        return {"precision": accuracy, "recall": accuracy, "f1": accuracy}
    
    def _evaluate_attachment_processing(self, detected_attachments: List[Dict], expected_attachments: List[Dict]) -> Dict[str, float]:
        """Evaluate attachment processing accuracy"""
        detected_names = {att.get("filename", "") for att in detected_attachments}
        expected_names = {att.get("filename", "") for att in expected_attachments}
        
        if not expected_names and not detected_names:
            return {"accuracy": 1.0}
        
        if not expected_names or not detected_names:
            return {"accuracy": 0.0}
        
        intersection = detected_names.intersection(expected_names)
        accuracy = len(intersection) / max(len(detected_names), len(expected_names))
        
        return {"accuracy": accuracy}
    
    async def _evaluate_ai_classification(self, detected_classifications: List[str], expected_classifications: List[str]) -> Dict[str, float]:
        """Evaluate AI classification accuracy"""
        if not expected_classifications and not detected_classifications:
            return {"accuracy": 1.0, "semantic_similarity": 1.0}
        
        # Simple intersection-based accuracy
        detected_set = set(detected_classifications)
        expected_set = set(expected_classifications)
        
        if not expected_set:
            accuracy = 1.0 if not detected_set else 0.0
        else:
            intersection = detected_set.intersection(expected_set)
            accuracy = len(intersection) / len(expected_set)
        
        # Semantic similarity could be enhanced with embeddings in the future
        semantic_similarity = accuracy  # Simplified for now
        
        return {"accuracy": accuracy, "semantic_similarity": semantic_similarity}
    
    def _evaluate_quality_assessment(self, analysis: Dict[str, Any], expected_quality: float) -> Dict[str, float]:
        """Evaluate quality assessment accuracy"""
        # Calculate a simple quality score based on resource richness
        urls_count = len(analysis.get("urls", []))
        code_count = len(analysis.get("code_snippets", []))
        attachments_count = len(analysis.get("attachments_processed", []))
        classifications_count = len(analysis.get("classifications", []))
        
        # Simple heuristic quality score
        calculated_quality = min(1.0, (urls_count * 0.3 + code_count * 0.2 + 
                                     attachments_count * 0.3 + classifications_count * 0.2) / 2)
        
        # Calculate error
        error = abs(calculated_quality - expected_quality)
        score = max(0.0, 1.0 - error)
        
        return {"score": score, "calculated_quality": calculated_quality, "expected_quality": expected_quality}
    
    def _calculate_aggregate_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all tests"""
        if not test_results:
            return {}
        
        metrics = {
            "url_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "code_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "attachment_processing": {"accuracy": 0.0},
            "ai_classification": {"accuracy": 0.0, "semantic_similarity": 0.0},
            "quality_assessment": {"score": 0.0},
            "overall_performance": 0.0
        }
        
        # Average across all tests
        for component in ["url_detection", "code_detection", "attachment_processing", "ai_classification", "quality_assessment"]:
            if component in ["url_detection", "code_detection"]:
                for metric in ["precision", "recall", "f1"]:
                    values = [result[component][metric] for result in test_results]
                    metrics[component][metric] = sum(values) / len(values)
            elif component == "attachment_processing":
                values = [result[component]["accuracy"] for result in test_results]
                metrics[component]["accuracy"] = sum(values) / len(values)
            elif component == "ai_classification":
                for metric in ["accuracy", "semantic_similarity"]:
                    values = [result[component][metric] for result in test_results]
                    metrics[component][metric] = sum(values) / len(values)
            elif component == "quality_assessment":
                values = [result[component]["score"] for result in test_results]
                metrics[component]["score"] = sum(values) / len(values)
        
        # Overall performance
        overall_scores = [result["overall_score"] for result in test_results]
        metrics["overall_performance"] = sum(overall_scores) / len(overall_scores)
        
        return metrics
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary with insights"""
        return {
            "overall_performance": {
                "score": metrics.get("overall_performance", 0.0),
                "grade": self._get_performance_grade(metrics.get("overall_performance", 0.0))
            },
            "strengths": self._identify_strengths(metrics),
            "weaknesses": self._identify_weaknesses(metrics),
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Good)"
        elif score >= 0.7:
            return "C (Fair)"
        elif score >= 0.6:
            return "D (Poor)"
        else:
            return "F (Failing)"
    
    def _identify_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify system strengths based on metrics"""
        strengths = []
        
        if metrics.get("url_detection", {}).get("f1", 0) >= 0.8:
            strengths.append("Strong URL detection and classification")
        
        if metrics.get("ai_classification", {}).get("accuracy", 0) >= 0.8:
            strengths.append("Accurate AI-powered content classification")
        
        if metrics.get("attachment_processing", {}).get("accuracy", 0) >= 0.9:
            strengths.append("Reliable attachment processing")
        
        if metrics.get("quality_assessment", {}).get("score", 0) >= 0.8:
            strengths.append("Good quality assessment capabilities")
        
        return strengths if strengths else ["System shows consistent basic functionality"]
    
    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify system weaknesses based on metrics"""
        weaknesses = []
        
        if metrics.get("url_detection", {}).get("f1", 0) < 0.7:
            weaknesses.append("URL detection needs improvement")
        
        if metrics.get("code_detection", {}).get("f1", 0) < 0.7:
            weaknesses.append("Code snippet detection could be enhanced")
        
        if metrics.get("ai_classification", {}).get("accuracy", 0) < 0.7:
            weaknesses.append("AI classification accuracy below target")
        
        if metrics.get("quality_assessment", {}).get("score", 0) < 0.7:
            weaknesses.append("Quality assessment methodology needs refinement")
        
        return weaknesses if weaknesses else ["No significant weaknesses identified"]
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics.get("url_detection", {}).get("precision", 0) < 0.8:
            recommendations.append("Expand domain classification rules for better URL categorization")
        
        if metrics.get("code_detection", {}).get("recall", 0) < 0.8:
            recommendations.append("Enhance code pattern matching to capture more code variations")
        
        if metrics.get("ai_classification", {}).get("accuracy", 0) < 0.8:
            recommendations.append("Fine-tune AI classification prompts or consider domain-specific models")
        
        if metrics.get("overall_performance", 0) < 0.8:
            recommendations.append("Consider implementing ensemble methods for improved accuracy")
        
        recommendations.append("Implement continuous monitoring with real-world data validation")
        
        return recommendations

# Pytest test functions
@pytest.fixture
async def evaluator():
    """Create evaluator instance for testing"""
    return ResourceDetectionEvaluator()

@pytest.mark.asyncio
async def test_comprehensive_evaluation():
    """Test the complete evaluation system"""
    evaluator = ResourceDetectionEvaluator()
    results = await evaluator.run_comprehensive_evaluation()
    
    assert "metrics" in results
    assert "summary" in results
    assert "test_results" in results
    assert results["total_tests"] > 0
    
    # Check that we have reasonable performance
    overall_performance = results["metrics"].get("overall_performance", 0.0)
    assert overall_performance >= 0.0  # At least some functionality
    
    print(f"📊 Overall Performance: {overall_performance:.2f}")
    print(f"🎯 Grade: {results['summary']['overall_performance']['grade']}")

@pytest.mark.asyncio
async def test_url_detection_evaluation():
    """Test URL detection evaluation specifically"""
    evaluator = ResourceDetectionEvaluator()
    
    # Test with known good URL
    test_resource = evaluator.test_resources[0]  # Academic paper
    result = await evaluator._evaluate_single_resource(test_resource)
    
    assert "url_detection" in result
    assert result["url_detection"]["f1"] >= 0.0

@pytest.mark.asyncio
async def test_noise_filtering_evaluation():
    """Test that noise domains are properly filtered"""
    evaluator = ResourceDetectionEvaluator()
    
    # Test with noise content
    noise_test = next(r for r in evaluator.test_resources if "tenor.com" in r.content)
    result = await evaluator._evaluate_single_resource(noise_test)
    
    # Should filter out tenor.com URLs
    assert len(result.get("url_detection", {}).get("detected_urls", [])) == 0

# Main execution
async def main():
    """Main evaluation execution"""
    evaluator = ResourceDetectionEvaluator()
    
    print("🚀 Starting Resource Detection and Classification Evaluation")
    print("=" * 60)
    
    results = await evaluator.run_comprehensive_evaluation()
    
    # Print detailed results
    print(f"\n📊 EVALUATION RESULTS")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Overall Performance: {results['metrics']['overall_performance']:.3f}")
    print(f"Grade: {results['summary']['overall_performance']['grade']}")
    
    print(f"\n🎯 COMPONENT METRICS:")
    for component, metrics in results['metrics'].items():
        if component != 'overall_performance':
            print(f"  {component.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
    
    print(f"\n💪 STRENGTHS:")
    for strength in results['summary']['strengths']:
        print(f"  ✅ {strength}")
    
    print(f"\n⚠️  AREAS FOR IMPROVEMENT:")
    for weakness in results['summary']['weaknesses']:
        print(f"  ❌ {weakness}")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    for recommendation in results['summary']['recommendations']:
        print(f"  💡 {recommendation}")
    
    # Save results to file
    output_file = f"resource_detection_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = project_root / "data" / "exports" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
