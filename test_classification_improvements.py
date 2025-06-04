#!/usr/bin/env python3
"""
Test the improved AI classification system directly
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from agentic.services.content_processor import ContentProcessingService

# Load environment variables
load_dotenv()

async def test_classification_improvements():
    """Test the improved classification system with real Discord content examples"""
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize content processor
    processor = ContentProcessingService(openai_client)
    
    # Test cases representing different types of Discord content
    test_cases = [
        {
            "name": "High-Quality Resource Sharing",
            "content": "Check out this new paper on transformer architecture improvements: https://arxiv.org/abs/2023.12345 - they achieved 15% better performance with novel attention mechanism. The key innovation is in the multi-head attention computation."
        },
        {
            "name": "Code Help Question",
            "content": "I'm getting a CUDA out of memory error when training my model. Here's my code: ```python\nmodel = GPT2Model(config)\noptimizer = torch.optim.Adam(model.parameters())\n```\nAny ideas on how to reduce memory usage?"
        },
        {
            "name": "Low Quality Meme",
            "content": "lol this AI meme is hilarious 😂"
        },
        {
            "name": "Educational Tutorial",
            "content": "Here's a step-by-step tutorial on implementing GPT from scratch. First, we'll cover attention mechanisms: the core idea is to compute weighted averages of input representations. Then we'll implement positional encoding to handle sequence order, and finally discuss training strategies including gradient clipping and learning rate scheduling."
        },
        {
            "name": "Technical Discussion",
            "content": "What do you think about the recent developments in multimodal AI? I've been reading about CLIP and DALL-E improvements, particularly how they handle the alignment between vision and language representations. The contrastive learning approach seems promising."
        },
        {
            "name": "Project Showcase",
            "content": "Just built a sentiment analysis app using BERT! Demo here: https://github.com/user/sentiment-app. Looking for feedback on the architecture - I used FastAPI for the backend and implemented custom preprocessing for social media text."
        },
        {
            "name": "Simple Question",
            "content": "Quick question - what's the difference between batch norm and layer norm?"
        },
        {
            "name": "General Greeting",
            "content": "Morning everyone! Hope you have a great day"
        },
        {
            "name": "Spam/Low Quality",
            "content": "hey"
        },
        {
            "name": "Tech Tip Tuesday Example",
            "content": "Tech Tip Tuesday! **Headline:** I test ChatGPT for a living — 7 secrets to instantly up your prompt game **Source:** Tom's Guide, April 26, 2025 **Link:** https://www.tomsguide.com/ai/chatgpt **Summary:** Professional tips for better AI prompting including role assignment, chain-of-thought reasoning, and iterative refinement. **Action:** Try assigning ChatGPT a specific role in your next prompt."
        }
    ]
    
    print("🧪 Testing Improved AI Classification System\n")
    print("=" * 80)
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        print(f"Content: {test_case['content'][:100]}...")
        
        try:
            # Test AI classification
            ai_classifications = await processor._classify_content_ai(test_case['content'])
            print(f"AI Classifications: {ai_classifications}")
            
            # Test heuristic fallback
            heuristic_classifications = processor._classify_content_heuristic(test_case['content'])
            print(f"Heuristic Classifications: {heuristic_classifications}")
            
            # Test full analysis
            mock_message = {
                'content': test_case['content'],
                'message_id': f'test_{i}',
                'timestamp': '2025-06-05T00:00:00Z',
                'author': {'username': 'test_user'},
                'channel_name': 'test_channel',
                'attachments': []
            }
            
            full_analysis = await processor.analyze_message_content(mock_message)
            # Add content to analysis for quality assessment
            full_analysis['content'] = test_case['content']
            
            quality_score = processor.calculate_quality_score(full_analysis)
            should_include = processor.should_include_resource(full_analysis)
            
            print(f"Quality Score: {quality_score:.3f}")
            print(f"Should Include: {should_include}")
            print(f"Content Length: {len(test_case['content'])}")
            print(f"Classifications: {full_analysis.get('classifications', [])}")
            
            results.append({
                'name': test_case['name'],
                'content_length': len(test_case['content']),
                'ai_classifications': ai_classifications,
                'heuristic_classifications': heuristic_classifications,
                'quality_score': quality_score,
                'should_include': should_include
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 CLASSIFICATION IMPROVEMENT TEST SUMMARY")
    print("=" * 80)
    
    included_count = sum(1 for r in results if r.get('should_include', False))
    total_count = len([r for r in results if 'error' not in r])
    
    print(f"Resources that would be included: {included_count}/{total_count}")
    if total_count > 0:
        print(f"Average quality score: {sum(r.get('quality_score', 0) for r in results if 'error' not in r) / total_count:.3f}")
    
    print("\nQuality filtering results:")
    for result in results:
        if 'error' not in result:
            status = "✅ INCLUDED" if result['should_include'] else "❌ FILTERED"
            print(f"  {result['name']}: {status} (score: {result['quality_score']:.3f})")
    
    # Test specific improvements
    print("\n🎯 Key Improvements Tested:")
    
    # Check if low quality content is properly filtered
    low_quality_filtered = all(
        not r.get('should_include', True) 
        for r in results 
        if r.get('name') in ['Low Quality Meme', 'Spam/Low Quality'] and 'error' not in r
    )
    print(f"  ✅ Low quality filtering: {'WORKING' if low_quality_filtered else 'NEEDS WORK'}")
    
    # Check if high quality content is included
    high_quality_included = all(
        r.get('should_include', False)
        for r in results 
        if r.get('name') in ['High-Quality Resource Sharing', 'Educational Tutorial', 'Tech Tip Tuesday Example'] and 'error' not in r
    )
    print(f"  ✅ High quality inclusion: {'WORKING' if high_quality_included else 'NEEDS WORK'}")
    
    # Check quality score distribution
    quality_scores = [r.get('quality_score', 0) for r in results if 'error' not in r]
    if quality_scores:
        max_score = max(quality_scores)
        min_score = min(quality_scores)
        print(f"  ✅ Quality score range: {min_score:.3f} - {max_score:.3f}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_classification_improvements())
