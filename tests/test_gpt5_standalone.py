#!/usr/bin/env python3
"""
Standalone test for GPT-5 and enrichment services
Imports directly without going through agentic/__init__.py
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid chromadb dependency
from agentic.services.gpt5_service import GPT5Service
from agentic.services.web_scraper import WebScraper
from agentic.services.resource_enrichment import ResourceEnrichment


async def test_gpt5():
    print("\n" + "="*80)
    print("🤖 TEST 1: GPT-5 Service")
    print("="*80)
    
    gpt5 = GPT5Service(use_cache=True)
    
    if gpt5.api_key:
        print("✅ OpenAI API key found in .env")
    else:
        print("⚠️ No API key - will use local LLM fallback")
    
    prompt = "In 10 words or less, what is artificial intelligence?"
    print(f"\n📝 Test prompt: {prompt}")
    
    try:
        response = await gpt5.generate(prompt, temperature=1.0, max_tokens=50)  # GPT-5-mini requires default temperature
        print(f"✅ Response: {response}")
        
        stats = gpt5.get_stats()
        print(f"\n📊 Stats:")
        print(f"   GPT-5 calls: {stats['gpt5_calls']}")
        print(f"   Cached: {stats['gpt5_cached']}")
        print(f"   Fallback: {stats['fallback_calls']}")
        print(f"   Errors: {stats['errors']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_scraper():
    print("\n" + "="*80)
    print("🌐 TEST 2: Web Scraper")
    print("="*80)
    
    scraper = WebScraper()
    
    test_url = "https://openai.com"
    print(f"\n🔍 Scraping: {test_url}")
    
    try:
        metadata = await scraper.extract_metadata(test_url)
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Description: {metadata.get('description', 'N/A')[:100]}...")
        print(f"   Method: {metadata.get('extraction_method', 'N/A')}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def test_enrichment():
    print("\n" + "="*80)
    print("✨ TEST 3: Full Resource Enrichment")
    print("="*80)
    
    enrichment = ResourceEnrichment(use_gpt5=True)
    
    test_url = "https://github.com/anthropics/prompt-eng-interactive-tutorial"
    test_message = {
        "content": "Great interactive tutorial for learning prompt engineering!",
        "author": {"display_name": "TestUser"},
        "channel_name": "resources"
    }
    
    print(f"\n🔗 URL: {test_url}")
    print(f"💬 Context: {test_message['content']}")
    
    try:
        result = await enrichment.enrich_resource(
            url=test_url,
            message=test_message,
            channel_name="resources"
        )
        
        print(f"\n✅ RESULTS:")
        print(f"   Title: {result.get('title')}")
        print(f"   Description: {result.get('description')[:150]}...")
        
        stats = enrichment.get_stats()
        print(f"\n📊 Enrichment Stats:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Titles scraped: {stats['titles_scraped']}")
        print(f"   Titles generated: {stats['titles_generated']}")
        print(f"   Descriptions generated: {stats['descriptions_generated']}")
        
        if enrichment.gpt5:
            gpt5_stats = enrichment.gpt5.get_stats()
            print(f"\n💰 Cost Analysis:")
            print(f"   GPT-5 calls: {gpt5_stats['gpt5_calls']}")
            if gpt5_stats['gpt5_calls'] > 0:
                cost = gpt5_stats['gpt5_calls'] * 0.02
                print(f"   Estimated cost: ${cost:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n🚀 PHASE 1 & 2 STANDALONE TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("GPT-5 Service", await test_gpt5()))
    results.append(("Web Scraper", await test_scraper()))
    results.append(("Full Enrichment", await test_enrichment()))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! GPT-5 enrichment is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")


if __name__ == '__main__':
    asyncio.run(main())


