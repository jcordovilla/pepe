#!/usr/bin/env python3
"""
Demo Quality Test

A simplified version of the quality test for demonstration purposes.
This version uses mock responses to show how the evaluation system works.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI


class MockAgentOrchestrator:
    """Mock orchestrator for demonstration"""
    
    async def process_query(self, query: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Mock responses for different query types"""
        query_lower = query.lower()
        
        if "machine learning" in query_lower:
            return """
            Here are the recent discussions about machine learning:
            
            **Found 8 messages:**
            • John Smith (2 hours ago): "Just read an interesting paper on transformer architectures"
            • Sarah Johnson (4 hours ago): "Anyone working on ML projects this week?"
            • Mike Chen (1 day ago): "Check out this new PyTorch tutorial"
            
            **Key Topics:**
            - Transformer architectures
            - PyTorch tutorials  
            - Project collaboration
            
            **Most Active:** John Smith (3 messages), Sarah Johnson (2 messages)
            """
        
        elif "weekly digest" in query_lower:
            return """
            # 📊 Weekly Digest: June 8-15, 2025
            
            **Summary:** 234 messages from 18 active users
            
            ## 👥 Most Active Users
            • John Smith: 45 messages
            • Sarah Johnson: 32 messages
            • Mike Chen: 28 messages
            
            ## 📋 Channel Activity
            ### #general (89 messages)
            • High engagement discussion about AI developments
            • Project updates and announcements
            
            ### #ai-research (67 messages)  
            • Technical discussions on transformer models
            • Paper sharing and reviews
            
            ## 🔥 High Engagement Content
            • "Breakthrough in AGI research" - 12 reactions
            • "New project proposal" - 8 reactions
            """
        
        elif "last week" in query_lower:
            return """
            **Discussions from June 8-15, 2025:**
            
            Found 156 messages across 12 channels:
            
            **#general:** Project kickoff meetings, team updates
            **#ai-research:** Paper discussions, technical deep-dives  
            **#random:** Community chat, memes, casual conversations
            
            **Timeline:**
            • June 15: Project demo presentations
            • June 14: Technical architecture discussions
            • June 13: Planning sessions
            • June 12: Research paper reviews
            
            **Key Contributors:** John, Sarah, Mike, Alex
            """
        
        else:
            return f"""
            I searched for "{query}" but didn't find specific matching content.
            
            **Suggestions:**
            • Try more specific keywords
            • Check channel names with #channel-name
            • Use time ranges like "last week" or "yesterday"
            
            **Available commands:**
            • Search: "find messages about [topic]"
            • Digest: "weekly digest" or "summary of last week"
            • Channel: "activity in #channel-name"
            """


class SimpleResponseQualityDemo:
    """Simplified demo of the quality evaluation system"""
    
    def __init__(self):
        self.orchestrator = MockAgentOrchestrator()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        self.demo_queries = [
            {
                "query": "Find messages about machine learning",
                "category": "search",
                "expected_score": 8.0
            },
            {
                "query": "Give me a weekly digest",
                "category": "digest", 
                "expected_score": 8.5
            },
            {
                "query": "Show me discussions from last week",
                "category": "temporal",
                "expected_score": 7.5
            },
            {
                "query": "What's the weather like?",
                "category": "out_of_scope",
                "expected_score": 5.0
            }
        ]
    
    async def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """Simple evaluation using OpenAI"""
        try:
            prompt = f"""
            Evaluate this Discord bot response on a scale of 1-10 for:
            1. Accuracy (factual correctness)
            2. Relevance (answers the question)
            3. Usefulness (provides value)
            
            Query: {query}
            Response: {response}
            
            Return JSON: {{"accuracy": X, "relevance": X, "usefulness": X, "overall": X, "comment": "brief comment"}}
            """
            
            completion = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            response_text = completion.choices[0].message.content
            if response_text:
                return json.loads(response_text)
            else:
                return {"accuracy": 5, "relevance": 5, "usefulness": 5, "overall": 5, "comment": "No evaluation"}
                
        except Exception as e:
            return {"accuracy": 0, "relevance": 0, "usefulness": 0, "overall": 0, "comment": f"Error: {e}"}
    
    async def run_demo(self):
        """Run the demo evaluation"""
        print("🚀 Agent Response Quality Demo")
        print("="*50)
        
        results = []
        
        for test_case in self.demo_queries:
            query = test_case["query"]
            category = test_case["category"]
            
            print(f"\n📝 Testing: {query}")
            print(f"Category: {category}")
            
            # Get response from mock agent
            response = await self.orchestrator.process_query(query, "demo_user")
            
            print(f"\n🤖 Agent Response:")
            print(response[:200] + "..." if len(response) > 200 else response)
            
            # Evaluate response quality (if OpenAI key available)
            if os.getenv('OPENAI_API_KEY'):
                evaluation = await self.evaluate_response(query, response)
                print(f"\n📊 AI Evaluation:")
                print(f"  Overall Score: {evaluation['overall']}/10")
                print(f"  Accuracy: {evaluation['accuracy']}/10")
                print(f"  Relevance: {evaluation['relevance']}/10") 
                print(f"  Usefulness: {evaluation['usefulness']}/10")
                print(f"  Comment: {evaluation['comment']}")
                
                results.append({
                    "query": query,
                    "category": category,
                    "score": evaluation['overall'],
                    "expected": test_case["expected_score"]
                })
            else:
                print("\n⚠️  OpenAI API key not set - skipping AI evaluation")
                results.append({
                    "query": query,
                    "category": category, 
                    "score": 0,
                    "expected": test_case["expected_score"]
                })
            
            print("-" * 50)
        
        # Summary
        if results and any(r["score"] > 0 for r in results):
            avg_score = sum(r["score"] for r in results if r["score"] > 0) / len([r for r in results if r["score"] > 0])
            print(f"\n🎯 Demo Summary:")
            print(f"Average Score: {avg_score:.1f}/10")
            print(f"Tests Run: {len(results)}")
            
            if avg_score >= 7.0:
                print("✅ Quality looks good!")
            elif avg_score >= 5.0:
                print("⚠️  Some improvement needed")
            else:
                print("❌ Significant issues detected")
        
        print("\n✨ Demo completed!")
        return results


async def main():
    """Run the demo"""
    demo = SimpleResponseQualityDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
