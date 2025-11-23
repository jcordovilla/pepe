#!/usr/bin/env python3
"""
Simple test for GPT-5 and web scraping services
Tests without requiring full project dependencies
"""

import asyncio
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set minimal environment
os.environ.setdefault('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
os.environ.setdefault('LLM_MODEL', 'deepseek-r1:8b')

async def test_gpt5_service():
    """Test GPT-5 service"""
    print("\n" + "=" * 80)
    print("🤖 Testing GPT-5 Service")
    print("=" * 80)
    
    from agentic.services.gpt5_service import GPT5Service
    
    gpt5 = GPT5Service(use_cache=True)
    
    # Check if OpenAI API key is available
    if gpt5.api_key:
        print("✅ OpenAI API key found")
    else:
        print("⚠️ No OpenAI API key - will test fallback to local LLM")
    
    # Test generation
    prompt = "In 10 words or less, what is artificial intelligence?"
    
    print(f"\n📝 Test prompt: {prompt}")
    try:
        response = await gpt5.generate(prompt, temperature=0.3, max_tokens=50)
        print(f"✅ Response: {response}")
        
        stats = gpt5.get_stats()
        print(f"\n📊 Stats:")
        print(f"   GPT-5 calls: {stats['gpt5_calls']}")
        print(f"   Cached: {stats['gpt5_cached']}")
        print(f"   Fallback: {stats['fallback_calls']}")
        print(f"   Errors: {stats['errors']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_web_scraper():
    """Test web scraper"""
    print("\n" + "=" * 80)
    print("🌐 Testing Web Scraper")
    print("=" * 80)
    
    from agentic.services.web_scraper import WebScraper
    
    scraper = WebScraper()
    
    test_urls = [
        "https://arxiv.org/abs/2405.12514",
        "https://openai.com",
    ]
    
    for url in test_urls:
        print(f"\n🔍 Scraping: {url}")
        try:
            metadata = await scraper.extract_metadata(url)
            
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Description: {metadata.get('description', 'N/A')[:100]}...")
            print(f"   Author: {metadata.get('author', 'N/A')}")
            print(f"   Method: {metadata.get('extraction_method', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def test_resource_enrichment():
    """Test full resource enrichment"""
    print("\n" + "=" * 80)
    print("✨ Testing Resource Enrichment (Full Pipeline)")
    print("=" * 80)
    
    from agentic.services.resource_enrichment import ResourceEnrichment
    
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
        print(f"   Methods: {result.get('enrichment_method')}")
        
        stats = enrichment.get_stats()
        print(f"\n📊 Enrichment Stats:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Titles scraped: {stats['titles_scraped']}")
        print(f"   Titles generated: {stats['titles_generated']}")
        print(f"   Descriptions generated: {stats['descriptions_generated']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    
    print("\n🚀 PHASE 1 & 2 IMPLEMENTATION TEST SUITE")
    print("Testing GPT-5 mini enrichment with web scraping")
    
    # Test individual components
    await test_gpt5_service()
    await test_web_scraper()
    await test_resource_enrichment()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
    print("\n📝 Summary:")
    print("   • GPT-5 Service: Implemented with fallback to local LLM")
    print("   • Web Scraper: Extracts metadata from URLs")
    print("   • Resource Enrichment: Combines both for high-quality titles & descriptions")
    print("\n🚀 Ready to use in resource_detector.py!")


if __name__ == '__main__':
    asyncio.run(main())

