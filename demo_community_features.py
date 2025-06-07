#!/usr/bin/env python3
"""
Comprehensive demonstration of community-enhanced Discord bot features
"""

import sys
import os
sys.path.append('/Users/jose/Documents/apps/discord-bot')

from tools.tools import (
    search_messages,
    find_community_experts,
    search_conversation_threads,
    analyze_community_engagement,
    test_community_search
)

def demonstrate_community_features():
    """Comprehensive demonstration of community features."""
    print("🤖 Discord Bot Community Enhancement Demo")
    print("=" * 60)
    
    print("\n🔍 1. ENHANCED SEMANTIC SEARCH")
    print("-" * 40)
    results = search_messages(query="machine learning help", k=5)
    print(f"Query: 'machine learning help' - Found {len(results)} results")
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Author: {result['author']['username']}")
        print(f"   Channel: #{result['channel_name']}")
        print(f"   Content: {result['content'][:120]}...")
        print(f"   Time: {result['timestamp']}")
    
    print("\n🎯 2. EXPERT IDENTIFICATION")
    print("-" * 40)
    
    # Find Python experts
    python_experts = find_community_experts("python", k=5, min_expertise_score=0.3)
    print(f"Python Experts ({len(python_experts)} found):")
    for expert in python_experts[:3]:
        print(f"  • {expert['author_name']}")
        print(f"    Expertise Score: {expert['expertise_score']}")
        print(f"    Skill Mentions: {expert['skill_mentions']}")
        print(f"    Skills: {', '.join(expert['skills'][:5])}")
        if expert['example_messages']:
            print(f"    Example: {expert['example_messages'][0]['content'][:80]}...")
        print()
    
    # Find AI/ML experts
    ai_experts = find_community_experts("artificial intelligence", k=3, min_expertise_score=0.2)
    print(f"AI/ML Experts ({len(ai_experts)} found):")
    for expert in ai_experts:
        print(f"  • {expert['author_name']} (Score: {expert['expertise_score']})")
    
    print("\n💬 3. CONVERSATION THREADING")
    print("-" * 40)
    
    threads = search_conversation_threads("learning programming", k=5, min_participants=2)
    print(f"Programming Learning Discussions ({len(threads)} threads):")
    
    for i, thread in enumerate(threads[:3], 1):
        print(f"\n{i}. Thread in #{thread['channel_name']}")
        print(f"   Participants: {thread['participant_count']} ({', '.join(thread['participants'][:3])})")
        print(f"   Messages: {thread['message_count']}")
        print(f"   Duration: {thread['start_time']} to {thread['end_time']}")
        print(f"   Has Questions: {thread['has_questions']} | Has Solutions: {thread['has_solutions']}")
        print(f"   Resolved: {thread['question_resolved']}")
        print(f"   Summary: {thread['summary']}")
    
    print("\n📊 4. COMMUNITY ENGAGEMENT ANALYSIS")
    print("-" * 40)
    
    engagement = analyze_community_engagement(time_period_days=30)
    if engagement:
        print("Overall Community Health (Last 30 days):")
        print(f"  📨 Total Messages: {engagement['total_messages']:,}")
        print(f"  👥 Active Users: {engagement['unique_authors']}")
        print(f"  🧵 Conversation Threads: {engagement['conversation_threads']}")
        
        metrics = engagement['engagement_metrics']
        print(f"\nEngagement Breakdown:")
        print(f"  🙋 Help Seeking: {metrics['help_seeking_messages']}")
        print(f"  🤝 Help Providing: {metrics['help_providing_messages']}")
        print(f"  ❓ Questions Asked: {metrics['questions_asked']}")
        print(f"  ✅ Solutions Provided: {metrics['solutions_provided']}")
        print(f"  🔧 Technical Discussions: {metrics['technical_discussions']}")
        print(f"  ✅ Resolved Questions: {metrics['resolved_questions']}")
        print(f"  ❌ Unresolved Questions: {metrics['unresolved_questions']}")
        
        health = engagement['community_health']
        print(f"\nHealth Metrics:")
        print(f"  Help Ratio: {health['help_ratio']:.2f} (providing/seeking)")
        print(f"  Resolution Ratio: {health['resolution_ratio']:.2f}")
        print(f"  Messages per User: {health['engagement_per_author']:.1f}")
        
        print(f"\n🏆 Top Skills Discussed:")
        for skill, count in engagement['top_skills'][:8]:
            print(f"  • {skill}: {count} mentions")
        
        print(f"\n⭐ Top Contributors:")
        for contributor in engagement['top_contributors'][:5]:
            print(f"  • {contributor['author_name']}")
            print(f"    Contribution Score: {contributor['contribution_score']}")
            print(f"    Messages: {contributor['messages']} | Solutions: {contributor['solutions']}")
            print(f"    Avg Expertise: {contributor['avg_expertise']}")
    
    print("\n🔬 5. DETAILED COMMUNITY SEARCH")
    print("-" * 40)
    
    # Test different search types
    search_types = [
        ("general", "javascript tutorial"),
        ("experts", "database design"),
        ("threads", "debugging help"),
        ("engagement", "activity analysis")
    ]
    
    for search_type, query in search_types:
        print(f"\n{search_type.title()} Search: '{query}'")
        result = test_community_search(query, k=3, search_type=search_type)
        
        if result['status'] == 'success':
            if search_type == "general":
                print(f"  Found {result['results_count']} messages")
                if result['results']:
                    sample = result['results'][0]
                    features = sample['community_features']
                    print(f"  Sample: {sample['author_name']} - {sample['content'][:60]}...")
                    print(f"  Community Features: skills={len(features['skill_keywords'])}, "
                          f"expertise={features['expertise_confidence']:.2f}")
            
            elif search_type == "experts":
                print(f"  Found {result['results_count']} experts")
            
            elif search_type == "threads":
                print(f"  Found {result['results_count']} conversation threads")
            
            elif search_type == "engagement":
                if 'results' in result and result['results']:
                    eng_data = result['results']
                    print(f"  Analysis: {eng_data.get('total_messages', 0)} messages, "
                          f"{eng_data.get('unique_authors', 0)} users")
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")
    
    print("\n✅ Community Enhancement Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  🔍 Enhanced semantic search with community context")
    print("  🎯 Automatic expert identification and ranking")
    print("  💬 Conversation threading and resolution tracking")
    print("  📊 Comprehensive community engagement analytics")
    print("  🏷️ Skill detection and expertise mining")
    print("  🤝 Help-seeking/providing behavior analysis")

if __name__ == "__main__":
    demonstrate_community_features()
