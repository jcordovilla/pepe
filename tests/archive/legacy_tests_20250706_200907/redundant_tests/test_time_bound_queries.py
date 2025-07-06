import asyncio
import os
import sys
sys.path.append('.')

from agentic.reasoning.query_analyzer import QueryAnalyzer

def test_time_bound_queries():
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    analyzer = QueryAnalyzer({"llm_complexity_threshold": 0.85})
    analysis = asyncio.run(analyzer.analyze("show messages from last week"))
    assert analysis['grouped_entities'].get('time_range') is not None
    tr = analysis['grouped_entities']['time_range']
    assert 'start' in tr and 'end' in tr

if __name__ == "__main__":
    asyncio.run(test_time_bound_queries())
