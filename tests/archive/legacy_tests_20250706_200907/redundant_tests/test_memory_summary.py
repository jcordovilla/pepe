import asyncio
import os
import pytest

from agentic.memory.conversation_memory import ConversationMemory


@pytest.mark.asyncio
async def test_history_summarization(tmp_path):
    db_path = tmp_path / "memory.db"
    memory = ConversationMemory({"db_path": str(db_path), "max_history_length": 3})

    for i in range(5):
        await memory.add_interaction("u1", f"q{i}", f"r{i}")

    history = await memory.get_history("u1")
    assert isinstance(history, list)
    assert "summary" in history[0]
    assert len(history) == 4  # summary + last 3 interactions

    history2 = await memory.get_history("u1", summarize=True)
    assert "summary" in history2[0]
