import json
from typing import List, Dict, Any
from datetime import datetime
from rag_engine import get_agent_answer
from tools import get_channels

# Core test queries organized by functionality
TEST_QUERIES = [
    # Time-based queries (testing time parsing)
    {
        "category": "Time parsing",
        "queries": [
            "Show me messages from the last 3 hours",
            "List messages from the past 2 days",
            "What happened during last week",
            "Show activity since yesterday"
        ]
    },
    # Channel-specific queries (testing channel filtering)
    {
        "category": "Channel filtering",
        "queries": [
            "What's new in the announcements channel",
            "Show me the latest messages from the papers-and-publications channel",
            "List all messages from the discord-managers channel"
        ]
    },
    # Content-based queries (testing semantic search)
    {
        "category": "Content search",
        "queries": [
            "Find messages about AI tools",
            "Show me discussions about meetings",
            "List messages mentioning Zoom"
        ]
    },
    # Combined queries (testing multiple filters)
    {
        "category": "Combined filters",
        "queries": [
            "Show me messages about AI tools in the papers-and-publications channel from the last week",
            "List all posts by oscarsan.chez in the announcements channel today",
            "What has been discussed about meetings in the general channel in the past 2 days"
        ]
    }
]

def run_query_test(query: str, channel_id: int = None) -> Dict[str, Any]:
    """
    Run a single query test and return the results.
    """
    start_time = datetime.now()
    try:
        result = get_agent_answer(query, channel_id)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "query": query,
            "channel_id": channel_id,
            "result": result,
            "duration": duration,
            "success": True,
            "error": None
        }
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "query": query,
            "channel_id": channel_id,
            "result": None,
            "duration": duration,
            "success": False,
            "error": str(e)
        }

def run_test_suite() -> List[Dict[str, Any]]:
    """
    Run the entire test suite and return results.
    """
    results = []
    
    # Get available channels
    channels = get_channels()
    
    # Run each query category
    for category in TEST_QUERIES:
        print(f"\nTesting {category['category']}...")
        
        # Run queries without channel filter
        for query in category["queries"]:
            print(f"Running query: {query}")
            result = run_query_test(query)
            results.append(result)
        
        # Run one query per category with channel filter
        if channels:
            channel = channels[0]  # Use first channel for testing
            channel_id = channel["id"]
            channel_name = channel["name"]
            
            # Add channel context to first query in category
            channel_query = f"{category['queries'][0]} in the {channel_name} channel"
            print(f"Running query: {channel_query}")
            result = run_query_test(channel_query, channel_id)
            results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], filename: str = "query_test_results.json"):
    """
    Save test results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "successful_queries": sum(1 for r in results if r["success"]),
            "failed_queries": sum(1 for r in results if not r["success"]),
            "average_duration": sum(r["duration"] for r in results) / len(results),
            "results": results
        }, f, indent=2)

def analyze_results(results: List[Dict[str, Any]]):
    """
    Print analysis of test results.
    """
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    avg_duration = sum(r["duration"] for r in results) / total
    
    print("\nTest Results Analysis:")
    print(f"Total queries: {total}")
    print(f"Successful queries: {successful}")
    print(f"Failed queries: {failed}")
    print(f"Success rate: {(successful/total)*100:.2f}%")
    print(f"Average query duration: {avg_duration:.2f} seconds")
    
    if failed > 0:
        print("\nFailed queries:")
        for r in results:
            if not r["success"]:
                print(f"- Query: {r['query']}")
                print(f"  Error: {r['error']}")

if __name__ == "__main__":
    print("Starting query test suite...")
    results = run_test_suite()
    save_results(results)
    analyze_results(results)
    print("\nTest results have been saved to query_test_results.json") 