#!/usr/bin/env python3
"""
Test script to evaluate embedding model performance on Discord content
"""

from core.ai_client import get_ai_client
import time
import numpy as np
from collections import Counter

def test_embedding_performance():
    """Test the current embedding model performance on different types of Discord content"""
    
    # Initialize the AI client
    ai_client = get_ai_client()
    
    # Test different types of Discord content
    test_texts = [
        'Hey everyone! How are you doing today?',  # Short conversational
        'I need help with Python async programming. Can someone explain how asyncio.gather() works?',  # Technical question
        '''Here's my code:
```python
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com') as response:
            return await response.json()
```
Any suggestions for improvement?''',  # Code block
        'Check out this amazing paper on transformer architectures: https://arxiv.org/abs/1706.03762 - it explains attention mechanisms really well and shows how they revolutionized NLP. The key insight is that self-attention allows the model to relate different positions in a sequence directly.',  # Academic/research
        '🎉 Welcome to our Discord server! Please read the rules in #rules and introduce yourself in #introductions. Feel free to ask questions in #help!',  # Server management
        'Has anyone tried the new OpenAI GPT-4 model? I heard it has better reasoning capabilities than previous versions. Would love to hear your experiences with it.',  # AI discussion
        'lol 😂',  # Very short
        'AI ethics is a fascinating topic. We need to consider bias, fairness, transparency, privacy, and accountability when developing AI systems. What are your thoughts on establishing ethical guidelines for AI development?',  # Ethics discussion
    ]

    print('=== Embedding Performance Test ===')
    print(f'Model: {ai_client.config.models.embedding_model}')
    print(f'Dimension: {ai_client.config.models.embedding_dimension}')
    print()

    # Test individual embeddings
    individual_times = []
    for i, text in enumerate(test_texts):
        start_time = time.time()
        embedding = ai_client.create_embeddings(text)
        end_time = time.time()
        
        duration = (end_time - start_time) * 1000
        individual_times.append(duration)
        
        print(f'Text {i+1} ({len(text)} chars): {duration:.1f}ms')
        print(f'  Preview: {text[:60]}...')
        print(f'  Embedding shape: {embedding.shape}')
        print()

    # Test batch processing
    print('=== Batch Processing Test ===')
    start_time = time.time()
    batch_embeddings = ai_client.create_embeddings(test_texts)
    end_time = time.time()

    batch_duration = (end_time - start_time) * 1000
    print(f'Batch of {len(test_texts)} texts: {batch_duration:.1f}ms')
    print(f'Batch embedding shape: {batch_embeddings.shape}')
    print(f'Per-text average (batch): {batch_duration/len(test_texts):.1f}ms')
    print(f'Per-text average (individual): {sum(individual_times)/len(individual_times):.1f}ms')
    print(f'Batch efficiency: {(sum(individual_times)/batch_duration):.1f}x faster')
    
    # Test semantic similarity
    print('\n=== Semantic Similarity Test ===')
    query_text = "How to use async programming in Python?"
    query_embedding = ai_client.create_embeddings(query_text)
    
    # Calculate similarities with test texts
    similarities = []
    for i, text in enumerate(test_texts):
        text_embedding = ai_client.create_embeddings(text)
        
        # Cosine similarity
        similarity = np.dot(query_embedding.flatten(), text_embedding.flatten()) / (
            np.linalg.norm(query_embedding.flatten()) * np.linalg.norm(text_embedding.flatten())
        )
        similarities.append((i, text[:60], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print(f'Query: "{query_text}"')
    print('Most similar texts:')
    for rank, (idx, preview, sim) in enumerate(similarities[:3]):
        print(f'  {rank+1}. Text {idx+1} (sim: {sim:.3f}): {preview}...')
    
    return {
        'model': ai_client.config.models.embedding_model,
        'dimension': ai_client.config.models.embedding_dimension,
        'individual_avg_ms': sum(individual_times) / len(individual_times),
        'batch_total_ms': batch_duration,
        'batch_efficiency': sum(individual_times) / batch_duration,
        'similarities': similarities
    }

if __name__ == "__main__":
    test_embedding_performance()
