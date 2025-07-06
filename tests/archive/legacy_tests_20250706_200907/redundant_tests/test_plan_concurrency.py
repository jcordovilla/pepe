import asyncio
import time
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('.'))

from agentic.agents.orchestrator import AgentOrchestrator
from agentic.agents.base_agent import BaseAgent, AgentRole, AgentState, SubTask, ExecutionPlan, TaskStatus, agent_registry

class DummyAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentRole.SEARCHER, {})

    async def process(self, state: AgentState) -> AgentState:
        await asyncio.sleep(0.1)
        subtask = state["current_subtask"]
        state["task_result"] = subtask.id
        return state

    def can_handle(self, task: SubTask) -> bool:
        return True

def test_concurrent_subtask_execution():
    async def run_test():
        agent_registry._agents = {}
        dummy = DummyAgent()
        agent_registry.register_agent(dummy)

        orchestrator = AgentOrchestrator({})

        subtasks = [
            SubTask(id=f"t{i}", description="test", agent_role=AgentRole.SEARCHER,
                    task_type="dummy", parameters={}, dependencies=[]) for i in range(3)
        ]
        plan = ExecutionPlan(id="p1", query="q", subtasks=subtasks)
        state = {"task_plan": plan, "search_results": [], "metadata": {}, "user_context": {}, "messages": []}

        start = time.perf_counter()
        result_state = await orchestrator._execute_plan_node(state)
        duration = time.perf_counter() - start

        assert all(sub.status == TaskStatus.COMPLETED for sub in plan.subtasks)
        assert result_state["metadata"]["execution_results"] == [sub.id for sub in subtasks]
        assert duration < 0.25

    asyncio.run(run_test())
