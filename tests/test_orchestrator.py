import asyncio
from agentic.agents.orchestrator import AgentOrchestrator
from agentic.config.modernized_config import get_modernized_config
from agentic.agents.base_agent import SubTask, AgentRole, TaskStatus
from datetime import datetime

async def test_synthesize_results_node():
    config = get_modernized_config()
    orchestrator = AgentOrchestrator(config.get('orchestrator', {}))
    
    # Create a simple task plan
    subtask = SubTask(
        id='fallback_search_test',
        description='Search for relevant messages (fallback)',
        agent_role=AgentRole.SEARCHER,
        task_type='semantic_search',
        parameters={'query': 'hello', 'filters': {}, 'k': 10},
        dependencies=[],
        created_at=datetime.utcnow()
    )
    
    initial_state = {
        'messages': [{'role': 'user', 'content': 'hello'}],
        'user_context': {'query': 'hello', 'user_id': 'test'},
        'search_results': [
            {
                'content': 'Hello there!',
                'metadata': {'author': 'test_user'},
                'similarity': 0.8
            }
        ],
        'analysis_results': {},
        'errors': [],
        'metadata': {'start_time': 0, 'version': '1.0.0'},
        'response': None,
        'query_interpretation': {
            'intent': 'greeting',
            'entities': [{'type': 'keyword', 'value': 'hello', 'confidence': 0.95}],
            'subtasks': [],
            'confidence': 0.95,
            'rationale': 'Query is a simple greeting'
        },
        'task_plan': type('ExecutionPlan', (), {
            'subtasks': [subtask],
            'id': 'test_plan'
        })(),
        'current_step': 0,
        'current_subtask': None,
        'intent': None,
        'entities': None,
        'complexity_score': None,
        'execution_plan': None,
        'subtasks': None,
        'dependencies': None
    }
    
    print("Testing synthesize_results_node...")
    result = await orchestrator._synthesize_results_node(initial_state)
    print("Synthesize results node completed successfully")
    print("Result keys:", list(result.keys()))
    print("Response:", result.get('response', 'No response'))

if __name__ == "__main__":
    asyncio.run(test_synthesize_results_node()) 