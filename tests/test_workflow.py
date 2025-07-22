import asyncio
from langgraph.graph import StateGraph, END
from agentic.agents.base_agent import AgentState

def test_node(state):
    print('Test node executed')
    return state

async def test_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node('test', test_node)
    workflow.set_entry_point('test')
    workflow.add_edge('test', END)
    app = workflow.compile()
    
    initial_state = {
        'messages': [],
        'user_context': {'query': 'test'},
        'search_results': [],
        'analysis_results': {},
        'errors': [],
        'metadata': {},
        'response': None,
        'query_interpretation': {},
        'task_plan': None,
        'current_step': 0,
        'current_subtask': None,
        'intent': None,
        'entities': None,
        'complexity_score': None,
        'execution_plan': None,
        'subtasks': None,
        'dependencies': None
    }
    
    result = await app.ainvoke(initial_state, {'configurable': {'thread_id': 'test'}})
    print('Minimal workflow result:', list(result.keys()) if result else 'None')

if __name__ == "__main__":
    asyncio.run(test_workflow()) 