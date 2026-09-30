import pytest
import asyncio
from core import SubQuery
from core.decomposer import IntentDecomposer
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_decomposer_bypasses_simple_query():
    decomposer = IntentDecomposer()
    # Should bypass
    res = await decomposer.decompose("What is retrieval augmented generation?")
    assert len(res) == 1
    assert res[0].text == "What is retrieval augmented generation?"
    
@pytest.mark.asyncio
async def test_decomposer_bypasses_concise_query():
    decomposer = IntentDecomposer()
    # Should bypass
    res = await decomposer.decompose("Who was Vaswani?")
    assert len(res) == 1
    
@pytest.mark.asyncio
async def test_decomposer_handles_compound_query(monkeypatch):
    decomposer = IntentDecomposer()
    
    async def mock_litellm(*args, **kwargs):
        class MockChoice:
            class MockMessage:
                content = '{"sub_queries": [{"id": "sq_0", "text": "What is RAG?", "intent_type": "factual"}, {"id": "sq_1", "text": "What are causes of climate change?", "intent_type": "factual"}]}'
            message = MockMessage()
        class MockResponse:
            choices = [MockChoice()]
        return MockResponse()
        
    monkeypatch.setattr("litellm.acompletion", mock_litellm)
    
    # Should NOT bypass, uses LLM (mocked here or falls back)
    res = await decomposer.decompose("What is retrieval augmented generation, and what are the main causes of climate change?")
    assert len(res) == 2

@pytest.mark.asyncio
async def test_decomposer_handles_wait_query():
    decomposer = IntentDecomposer()
    # Should bypass
    res = await decomposer.decompose("um uh well so")
    assert len(res) == 1
    
@pytest.mark.asyncio
async def test_decomposer_handles_suppress_query():
    decomposer = IntentDecomposer()
    # Should bypass
    res = await decomposer.decompose("next slide please")
    assert len(res) == 1
