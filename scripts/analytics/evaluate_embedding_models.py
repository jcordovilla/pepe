#!/usr/bin/env python3
"""
Comprehensive evaluation of embedding models for Discord bot semantic search.
Tests multiple models against actual Discord content types to find optimal choice.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
from collections import defaultdict

# Test models - focusing on best local models for different use cases
TEST_MODELS = {
    # Current model
    'all-MiniLM-L6-v2': {
        'dimensions': 384,
        'description': 'Current model - balanced speed/quality',
        'category': 'general'
    },
    
    # High-quality general models
    'all-mpnet-base-v2': {
        'dimensions': 768,
        'description': 'High-quality general embedding model',
        'category': 'general-high'
    },
    
    'paraphrase-mpnet-base-v2': {
        'dimensions': 768,
        'description': 'Good for paraphrase detection and semantic similarity',
        'category': 'paraphrase'
    },
    
    # Multilingual models (important for Discord community)
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'dimensions': 384,
        'description': 'Multilingual model for diverse community',
        'category': 'multilingual'
    },
    
    'distiluse-base-multilingual-cased': {
        'dimensions': 512,
        'description': 'Multilingual distilled model',
        'category': 'multilingual'
    },
    
    # Fast/efficient models
    'all-MiniLM-L12-v2': {
        'dimensions': 384,
        'description': 'Larger version of current model',
        'category': 'fast'
    },
    
    # Domain-specific considerations
    'msmarco-distilbert-base-v4': {
        'dimensions': 768,
        'description': 'Optimized for search/retrieval tasks',
        'category': 'search'
    }
}

# Discord content samples representing different use cases
DISCORD_TEST_CONTENT = [
    # Short conversational
    "Hey everyone! How are you doing today?",
    "Thanks for sharing this!",
    "lol 😂",
    "Yes, exactly what I was thinking",
    
    # Technical questions/discussions
    "I need help with Python async programming. Can someone explain how asyncio.gather() works?",
    "Has anyone tried the new OpenAI GPT-4 model? I heard it has better reasoning capabilities.",
    "What's the best way to handle authentication in a REST API?",
    "How do you deploy machine learning models to production?",
    
    # Code snippets
    '''Here's my code:
```python
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com') as response:
            return await response.json()
```
Any suggestions for improvement?''',
    
    '''Check this out:
```javascript
const handleSubmit = async (data) => {
  try {
    const response = await fetch('/api/submit', {
      method: 'POST',
      body: JSON.stringify(data)
    });
    return response.json();
  } catch (error) {
    console.error(error);
  }
};
```''',
    
    # Academic/research content
    "Check out this amazing paper on transformer architectures: https://arxiv.org/abs/1706.03762 - it explains attention mechanisms really well.",
    "The key insight is that self-attention allows the model to relate different positions in a sequence directly.",
    "Recent research shows that larger models exhibit emergent capabilities that smaller models don't have.",
    
    # AI Ethics discussions (community focus)
    "AI ethics is a fascinating topic. We need to consider bias, fairness, transparency, privacy, and accountability when developing AI systems.",
    "What are your thoughts on establishing ethical guidelines for AI development?",
    "How do we ensure AI systems are developed responsibly?",
    "The impact of AI on employment is something we should discuss more.",
    
    # Community/server management
    "🎉 Welcome to our Discord server! Please read the rules in #rules and introduce yourself.",
    "Don't forget about our weekly meeting tomorrow at 3 PM UTC",
    "Please keep discussions in the appropriate channels",
    
    # Resource sharing
    "Here's a great tutorial on neural networks: https://example.com/tutorial",
    "I found this paper interesting: [link] - thoughts?",
    "Sharing some useful resources for beginners",
    
    # Multilingual content (community consideration)
    "Hola a todos! ¿Cómo están hoy?",
    "Bonjour, j'ai une question sur l'IA",
    "Hallo, kann jemand mir bei diesem Problem helfen?",
    "これはとても面白いですね",
    
    # Long-form content
    """This is a comprehensive explanation of how neural attention mechanisms work. 
    The attention mechanism allows models to focus on different parts of the input sequence 
    when generating each element of the output sequence. This is particularly useful in 
    sequence-to-sequence tasks like machine translation, where the model needs to align 
    parts of the source sentence with parts of the target sentence. The key innovation 
    of attention is that it provides a weighted combination of all input hidden states, 
    rather than just using the final hidden state from the encoder.""",
]

# Test queries for semantic similarity evaluation
TEST_QUERIES = [
    "Python programming help",
    "async await asyncio",
    "machine learning deployment",
    "AI ethics guidelines", 
    "transformer attention mechanisms",
    "Discord server rules",
    "code review suggestions",
    "research paper recommendations",
    "multilingual support",
    "neural network tutorials"
]

def load_model_safe(model_name: str) -> Tuple[SentenceTransformer, bool]:
    """Safely load a model, return None if failed"""
    try:
        print(f"Loading {model_name}...")
        model = SentenceTransformer(model_name)
        return model, True
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {str(e)}")
        return None, False

def benchmark_speed(model: SentenceTransformer, texts: List[str]) -> Dict[str, float]:
    """Benchmark embedding speed for a model"""
    
    # Individual processing
    individual_times = []
    for text in texts[:5]:  # Test first 5 for speed
        start = time.time()
        _ = model.encode(text)
        individual_times.append((time.time() - start) * 1000)
    
    # Batch processing
    start = time.time()
    _ = model.encode(texts[:10])  # Batch of 10
    batch_time = (time.time() - start) * 1000
    
    return {
        'avg_individual_ms': np.mean(individual_times),
        'batch_10_ms': batch_time,
        'batch_efficiency': np.mean(individual_times) * 10 / batch_time
    }

def evaluate_semantic_quality(model: SentenceTransformer, content: List[str], queries: List[str]) -> Dict[str, float]:
    """Evaluate semantic similarity quality"""
    
    # Encode all content and queries
    content_embeddings = model.encode(content)
    query_embeddings = model.encode(queries)
    
    # Calculate similarities
    similarities = np.dot(query_embeddings, content_embeddings.T)
    
    # Evaluate quality metrics
    results = {}
    
    # Average max similarity (how well does each query match its best content)
    max_similarities = np.max(similarities, axis=1)
    results['avg_max_similarity'] = np.mean(max_similarities)
    
    # Distribution analysis
    results['similarity_std'] = np.std(similarities.flatten())
    results['similarity_range'] = np.max(similarities) - np.min(similarities)
    
    # Specific test cases
    specific_tests = {
        'python_async': ('Python programming help', 'I need help with Python async programming'),
        'ai_ethics': ('AI ethics guidelines', 'AI ethics is a fascinating topic'),
        'code_help': ('code review suggestions', 'Any suggestions for improvement?')
    }
    
    for test_name, (query, expected_content) in specific_tests.items():
        if query in queries and expected_content in content:
            query_idx = queries.index(query)
            content_idx = content.index(expected_content)
            results[f'specific_{test_name}'] = similarities[query_idx][content_idx]
    
    return results

def evaluate_model(model_name: str, model_info: Dict) -> Dict:
    """Comprehensive evaluation of a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Description: {model_info['description']}")
    print(f"Dimensions: {model_info['dimensions']}")
    print(f"{'='*60}")
    
    # Load model
    model, success = load_model_safe(model_name)
    if not success:
        return {'model_name': model_name, 'status': 'failed_to_load'}
    
    results = {
        'model_name': model_name,
        'status': 'success',
        'dimensions': model_info['dimensions'],
        'category': model_info['category'],
        'description': model_info['description']
    }
    
    try:
        # Speed benchmark
        print("📊 Running speed benchmarks...")
        speed_results = benchmark_speed(model, DISCORD_TEST_CONTENT)
        results.update(speed_results)
        print(f"  Individual avg: {speed_results['avg_individual_ms']:.1f}ms")
        print(f"  Batch efficiency: {speed_results['batch_efficiency']:.1f}x")
        
        # Semantic quality
        print("🎯 Evaluating semantic quality...")
        quality_results = evaluate_semantic_quality(model, DISCORD_TEST_CONTENT, TEST_QUERIES)
        results.update(quality_results)
        print(f"  Avg max similarity: {quality_results['avg_max_similarity']:.3f}")
        print(f"  Similarity range: {quality_results['similarity_range']:.3f}")
        
        # Memory usage (approximate)
        embedding_sample = model.encode("test")
        results['embedding_size_bytes'] = embedding_sample.nbytes
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        results['status'] = 'evaluation_failed'
        results['error'] = str(e)
    
    return results

def rank_models(results: List[Dict]) -> List[Dict]:
    """Rank models based on multiple criteria for Discord use case"""
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        return results
    
    # Scoring criteria (weights can be adjusted)
    criteria = {
        'speed_score': 0.3,      # Faster is better
        'quality_score': 0.4,    # Higher similarity is better  
        'efficiency_score': 0.2, # Batch efficiency
        'size_score': 0.1        # Smaller embeddings preferred for speed
    }
    
    # Normalize scores
    for result in successful_results:
        # Speed score (inverse of time)
        max_speed = max(r.get('avg_individual_ms', float('inf')) for r in successful_results)
        result['speed_score'] = max_speed / result.get('avg_individual_ms', max_speed)
        
        # Quality score
        result['quality_score'] = result.get('avg_max_similarity', 0)
        
        # Efficiency score  
        result['efficiency_score'] = min(result.get('batch_efficiency', 1), 50) / 50  # Cap at 50x
        
        # Size score (inverse of dimensions)
        max_dims = max(r.get('dimensions', 1) for r in successful_results)
        result['size_score'] = max_dims / result.get('dimensions', max_dims)
        
        # Overall score
        result['overall_score'] = sum(
            criteria[criterion] * result[score_key] 
            for criterion, score_key in [
                ('speed_score', 'speed_score'),
                ('quality_score', 'quality_score'),
                ('efficiency_score', 'efficiency_score'),
                ('size_score', 'size_score')
            ]
        )
    
    # Sort by overall score
    successful_results.sort(key=lambda x: x['overall_score'], reverse=True)
    
    return successful_results + [r for r in results if r['status'] != 'success']

def main():
    """Run comprehensive embedding model evaluation"""
    print("🔍 Discord Bot Embedding Model Evaluation")
    print("=" * 60)
    print(f"Testing {len(TEST_MODELS)} models on {len(DISCORD_TEST_CONTENT)} content samples")
    print(f"Using {len(TEST_QUERIES)} test queries")
    
    all_results = []
    
    for model_name, model_info in TEST_MODELS.items():
        try:
            result = evaluate_model(model_name, model_info)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n⏹️ Evaluation interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error evaluating {model_name}: {str(e)}")
            all_results.append({
                'model_name': model_name,
                'status': 'unexpected_error',
                'error': str(e)
            })
    
    # Rank models
    ranked_results = rank_models(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    successful_models = [r for r in ranked_results if r['status'] == 'success']
    
    if successful_models:
        print(f"\n🏆 TOP RECOMMENDATIONS (out of {len(successful_models)} successful):")
        print("-" * 80)
        
        for i, result in enumerate(successful_models[:3], 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   Overall Score: {result['overall_score']:.3f}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Speed: {result.get('avg_individual_ms', 0):.1f}ms avg")
            print(f"   Quality: {result.get('avg_max_similarity', 0):.3f} similarity")
            print(f"   Batch Efficiency: {result.get('batch_efficiency', 0):.1f}x")
            print(f"   Category: {result['category']}")
            print(f"   Description: {result['description']}")
        
        print(f"\n📋 DETAILED COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<35} {'Dims':<6} {'Speed(ms)':<10} {'Quality':<9} {'Batch':<8} {'Score':<7}")
        print("-" * 80)
        
        for result in successful_models:
            print(f"{result['model_name']:<35} "
                  f"{result['dimensions']:<6} "
                  f"{result.get('avg_individual_ms', 0):<10.1f} "
                  f"{result.get('avg_max_similarity', 0):<9.3f} "
                  f"{result.get('batch_efficiency', 0):<8.1f} "
                  f"{result['overall_score']:<7.3f}")
    
    # Failed models
    failed_models = [r for r in all_results if r['status'] != 'success']
    if failed_models:
        print(f"\n❌ FAILED TO EVALUATE ({len(failed_models)} models):")
        for result in failed_models:
            print(f"   {result['model_name']}: {result['status']}")
    
    # Recommendation based on Discord use case
    print(f"\n🎯 RECOMMENDATION FOR DISCORD BOT:")
    print("-" * 50)
    
    if successful_models:
        best = successful_models[0]
        current = next((r for r in all_results if r['model_name'] == 'all-MiniLM-L6-v2'), None)
        
        print(f"Best Model: {best['model_name']}")
        print(f"Reason: {best['description']}")
        
        if current and current['status'] == 'success':
            if best['model_name'] != 'all-MiniLM-L6-v2':
                improvement_quality = (best.get('avg_max_similarity', 0) - current.get('avg_max_similarity', 0)) / current.get('avg_max_similarity', 1) * 100
                improvement_speed = (current.get('avg_individual_ms', 1) - best.get('avg_individual_ms', 1)) / current.get('avg_individual_ms', 1) * 100
                
                print(f"\nImprovement vs Current Model:")
                print(f"  Quality: {improvement_quality:+.1f}%")
                print(f"  Speed: {improvement_speed:+.1f}%")
                print(f"  Dimensions: {current['dimensions']} → {best['dimensions']}")
            else:
                print("\nCurrent model (all-MiniLM-L6-v2) is already optimal!")
    
    # Save detailed results
    with open('/Users/jose/Documents/apps/discord-bot/embedding_evaluation_results.json', 'w') as f:
        json.dump(ranked_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: embedding_evaluation_results.json")
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()
