#!/usr/bin/env python3
"""
Test script for Phase 1 & 2 Resource Enrichment
Tests title extraction and description generation with GPT-5 mini
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic.services.resource_enrichment import ResourceEnrichment

# Test URLs covering different domains
TEST_URLS = [
    {
        "url": "https://www.youtube.com/watch?v=JfE1Wun9xkk",
        "message": {
            "content": "Sam Altman discussing Sora, energy, and building an AI empire",
            "author": {"display_name": "TestUser"},
            "channel_name": "general-chat"
        },
        "expected_title_contains": "Sam Altman"
    },
    {
        "url": "https://arxiv.org/abs/2405.12514",
        "message": {
            "content": "Interesting paper about future self AI tool for reducing anxiety",
            "author": {"display_name": "TestUser"},
            "channel_name": "research"
        },
        "expected_title_contains": "arXiv"
    },
    {
        "url": "https://github.com/anthropics/prompt-eng-interactive-tutorial",
        "message": {
            "content": "Interactive tutorial for prompt engineering by Anthropic",
            "author": {"display_name": "TestUser"},
            "channel_name": "resources"
        },
        "expected_title_contains": "prompt"
    },
    {
        "url": "https://openai.com/index/introducing-chatgpt-agent",
        "message": {
            "content": "OpenAI just released ChatGPT Agent capabilities!",
            "author": {"display_name": "TestUser"},
            "channel_name": "genai-news"
        },
        "expected_title_contains": "ChatGPT"
    }
]


async def test_enrichment(use_gpt5: bool = True):
    """Test the resource enrichment service"""
    
    print("=" * 80)
    print(f"🧪 Testing Resource Enrichment (GPT-5: {'ENABLED' if use_gpt5 else 'DISABLED'})")
    print("=" * 80)
    
    enrichment = ResourceEnrichment(use_gpt5=use_gpt5)
    
    results = []
    
    for i, test_case in enumerate(TEST_URLS, 1):
        print(f"\n🔍 Test {i}/{len(TEST_URLS)}: {test_case['url']}")
        print(f"   Expected title contains: '{test_case['expected_title_contains']}'")
        
        try:
            enriched = await enrichment.enrich_resource(
                url=test_case['url'],
                message=test_case['message'],
                channel_name=test_case['message']['channel_name']
            )
            
            title = enriched.get('title')
            description = enriched.get('description')
            
            # Check if title is meaningful
            is_good_title = title and len(title) > 10 and title not in ['YouTube Video', 'Resource']
            expected_found = test_case['expected_title_contains'].lower() in title.lower() if title else False
            
            print(f"\n   ✅ Title: {title}")
            print(f"   📝 Description: {description[:150]}...")
            print(f"   🎯 Quality Check:")
            print(f"      - Title length: {len(title) if title else 0} chars")
            print(f"      - Description length: {len(description) if description else 0} chars")
            print(f"      - Title is meaningful: {'✅' if is_good_title else '❌'}")
            print(f"      - Contains expected keyword: {'✅' if expected_found else '⚠️'}")
            
            results.append({
                'url': test_case['url'],
                'title': title,
                'description': description,
                'is_good': is_good_title and len(description or '') > 50
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                'url': test_case['url'],
                'error': str(e),
                'is_good': False
            })
    
    # Display statistics
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    stats = enrichment.get_stats()
    print(f"\n📈 Enrichment Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   Titles scraped: {stats['titles_scraped']}")
    print(f"   Titles generated (GPT-5): {stats['titles_generated']}")
    print(f"   Descriptions generated: {stats['descriptions_generated']}")
    print(f"   Errors: {stats['errors']}")
    
    if enrichment.gpt5:
        gpt5_stats = enrichment.gpt5.get_stats()
        print(f"\n🤖 GPT-5 API Usage:")
        print(f"   API calls: {gpt5_stats['gpt5_calls']}")
        print(f"   Cached: {gpt5_stats['gpt5_cached']}")
        print(f"   Fallback to local LLM: {gpt5_stats['fallback_calls']}")
        
        if gpt5_stats['gpt5_calls'] > 0:
            estimated_cost = gpt5_stats['gpt5_calls'] * 0.02
            print(f"   Estimated cost: ${estimated_cost:.2f}")
    
    # Quality assessment
    good_results = [r for r in results if r.get('is_good', False)]
    success_rate = (len(good_results) / len(results)) * 100 if results else 0
    
    print(f"\n✨ Quality Assessment:")
    print(f"   Successful enrichments: {len(good_results)}/{len(results)}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print(f"   🎉 EXCELLENT! Implementation is working well.")
    elif success_rate >= 50:
        print(f"   👍 GOOD! Most resources enriched successfully.")
    else:
        print(f"   ⚠️ NEEDS IMPROVEMENT: Check errors above.")
    
    return results


async def compare_gpt5_vs_fallback():
    """Compare GPT-5 enrichment vs fallback methods"""
    
    print("\n" + "=" * 80)
    print("🔬 COMPARISON: GPT-5 Mini vs Fallback Methods")
    print("=" * 80)
    
    test_url = TEST_URLS[0]  # Use first test URL
    
    print("\n1️⃣ Testing with GPT-5 mini...")
    enrichment_gpt5 = ResourceEnrichment(use_gpt5=True)
    result_gpt5 = await enrichment_gpt5.enrich_resource(
        url=test_url['url'],
        message=test_url['message'],
        channel_name=test_url['message']['channel_name']
    )
    
    print("\n2️⃣ Testing with fallback methods...")
    enrichment_fallback = ResourceEnrichment(use_gpt5=False)
    result_fallback = await enrichment_fallback.enrich_resource(
        url=test_url['url'],
        message=test_url['message'],
        channel_name=test_url['message']['channel_name']
    )
    
    print("\n" + "-" * 80)
    print("COMPARISON RESULTS:")
    print("-" * 80)
    
    print(f"\n🤖 GPT-5 Mini:")
    print(f"   Title: {result_gpt5.get('title')}")
    print(f"   Description: {result_gpt5.get('description')[:200]}...")
    
    print(f"\n🔄 Fallback:")
    print(f"   Title: {result_fallback.get('title')}")
    print(f"   Description: {result_fallback.get('description')[:200]}...")
    
    print(f"\n📊 Quality Comparison:")
    gpt5_title_len = len(result_gpt5.get('title', ''))
    fallback_title_len = len(result_fallback.get('title', ''))
    gpt5_desc_len = len(result_gpt5.get('description', ''))
    fallback_desc_len = len(result_fallback.get('description', ''))
    
    print(f"   Title length: GPT-5 {gpt5_title_len} vs Fallback {fallback_title_len}")
    print(f"   Description length: GPT-5 {gpt5_desc_len} vs Fallback {fallback_desc_len}")
    print(f"   Winner: {'🤖 GPT-5' if gpt5_desc_len > fallback_desc_len else '🔄 Fallback'}")


async def main():
    """Run all tests"""
    
    print("\n🚀 Starting Resource Enrichment Tests")
    print("This will test Phase 1 (Title Extraction) and Phase 2 (Description Generation)\n")
    
    # Test with GPT-5
    try:
        results = await test_enrichment(use_gpt5=True)
    except Exception as e:
        print(f"\n❌ GPT-5 tests failed: {e}")
        print("Trying fallback methods...")
        results = await test_enrichment(use_gpt5=False)
    
    # Compare methods
    # await compare_gpt5_vs_fallback()
    
    print("\n✅ Tests completed!")
    print("\nNext steps:")
    print("  1. If OPENAI_API_KEY is set, you should see GPT-5 enrichment working")
    print("  2. If not, fallback methods will be used")
    print("  3. Run resource detector: python scripts/resource_detector.py")


if __name__ == '__main__':
    asyncio.run(main())

