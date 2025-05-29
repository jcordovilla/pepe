import json
from typing import List, Dict, Any
from datetime import datetime
from core.rag_engine import get_agent_answer

# Test queries organized by category based on actual database content
TEST_QUERIES = [
    {
        "category": "General Content Search",
        "queries": [
            "What did people discuss about AI ethics in the community?",
            "Find discussions about learning resources for non-coders",
            "What were the main topics in general chat recently?"
        ]
    },
    {
        "category": "Time-Bounded Queries",
        "queries": [
            "What was discussed in the last week about AI philosophy?",
            "Show me messages from yesterday about learning resources",
            "What were the main topics in April 2025?"
        ]
    },
    {
        "category": "Channel-Specific Queries",
        "queries": [
            "What was discussed in #general-chat about workshops?",
            "Show me messages from #ai-philosophy-ethics about AI governance",
            "What are the most common questions in #non-coders-learning?"
        ]
    },
    {
        "category": "Author/Member Queries",
        "queries": [
            "What has Irene Yang shared about learning resources?",
            "Show me all messages by Patrick in the last month",
            "What did Georgi contribute about AI ethics?"
        ]
    },
    {
        "category": "Learning Resources",
        "queries": [
            "Find discussions about Zoom meetings and workshops",
            "Show me messages about AI learning tools and platforms",
            "What resources were shared for non-coders?"
        ]
    },
    {
        "category": "Data Availability Queries",
        "queries": [
            "What data is currently available in the database?",
            "How many messages are in each channel?",
            "What channels have the most activity?"
        ]
    },
    {
        "category": "Resource/Link Discovery",
        "queries": [
            "Show me all shared Zoom meeting links",
            "What learning resources were posted in #non-coders-learning?",
            "Find AI tools and platforms mentioned in discussions"
        ]
    },
    {
        "category": "Summarization Queries",
        "queries": [
            "Summarize the main topics in #ai-philosophy-ethics last month",
            "Give me a summary of discussions about learning resources in the last week",
            "What were the key points about workshops in the past 2 days?"
        ]
    },
    {
        "category": "Jump to Message",
        "queries": [
            "Find the message where 'AI ethics' was first mentioned",
            "Show me the most recent message about 'workshops'",
            "Find the first discussion about learning resources"
        ]
    },
    {
        "category": "Channel/Server Structure",
        "queries": [
            "List all available channels and their message counts",
            "What is the channel ID for #general-chat?",
            "Which channels are related to learning and education?"
        ]
    }
]

def run_query_test(query: str) -> Dict[str, Any]:
    """
    Run a single query test and return the results.
    """
    try:
        response = get_agent_answer(query)
        return {
            "query": query,
            "response": response
        }
    except Exception as e:
        return {
            "query": query,
            "response": f"Error: {str(e)}"
        }

def run_test_suite() -> List[Dict[str, Any]]:
    """
    Run all queries and return results.
    """
    results = []
    
    # Run each query category
    for category in TEST_QUERIES:
        print(f"\nTesting {category['category']}...")
        
        # Run all queries in category
        for query in category["queries"]:
            print(f"Running query: {query}")
            result = run_query_test(query)
            results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], filename: str = "query_test_results.json"):
    """
    Save test results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    print("Starting query test suite...")
    results = run_test_suite()
    save_results(results)
    print("\nTest results have been saved to query_test_results.json") 