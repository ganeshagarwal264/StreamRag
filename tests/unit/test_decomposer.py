import pytest, json
from unittest.mock import patch, AsyncMock, MagicMock
from core import SubQuery
from core.decomposer import IntentDecomposer

@pytest.fixture
def decomposer():
    return IntentDecomposer()

def make_resp(data_obj):
    content = json.dumps(data_obj)
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp

@pytest.mark.asyncio
async def test_single_intent(decomposer):
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp({'sub_queries': [{'id': 'sq_0', 'text': 'What is RAG?', 'intent_type': 'factual', 'is_dependent': False}]})):
        r = await decomposer.decompose('What is RAG?')
    assert len(r) == 1
    assert r[0].intent_type == 'factual'

@pytest.mark.asyncio
async def test_multi_intent(decomposer):
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp({'sub_queries': [
                   {'id': 'sq_0', 'text': 'How do transformers work?', 'intent_type': 'factual', 'is_dependent': False},
                   {'id': 'sq_1', 'text': 'What is RAG?', 'intent_type': 'factual', 'is_dependent': False},
               ]})):
        r = await decomposer.decompose('How do transformers work and what is RAG?')
    assert len(r) == 2

@pytest.mark.asyncio
async def test_raw_array_format(decomposer):
    # Tests the fix for Gemma returning [...] directly instead of {"sub_queries": [...]}
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp([
                   {'id': 'sq_0', 'text': 'Question 1?', 'intent_type': 'factual', 'is_dependent': False},
                   {'id': 'sq_1', 'text': 'Question 2?', 'intent_type': 'factual', 'is_dependent': False},
               ])):
        r = await decomposer.decompose('Question 1 and question 2?')
    assert len(r) == 2

@pytest.mark.asyncio
async def test_fallback_on_bad_json(decomposer):
    bad = MagicMock()
    bad.choices = [MagicMock()]
    bad.choices[0].message.content = 'not json at all'
    with patch('litellm.acompletion', new_callable=AsyncMock, return_value=bad):
        r = await decomposer.decompose('What is ML?')
    assert len(r) == 1
    assert r[0].text == 'What is ML?'

@pytest.mark.asyncio
async def test_dependent_sub_query(decomposer):
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp({'sub_queries': [{'id': 'sq_0', 'text': 'What is X?', 'intent_type': 'factual', 'is_dependent': False},
                                       {'id': 'sq_1', 'text': 'How does X relate to Y?', 'intent_type': 'comparative', 'is_dependent': True}]})):
        r = await decomposer.decompose('What is X and how does it relate to Y?')
    assert r[1].is_dependent == True

@pytest.mark.asyncio
async def test_api_error_fallback(decomposer):
    with patch('litellm.acompletion', new_callable=AsyncMock, side_effect=Exception('API error')):
        r = await decomposer.decompose('Some query')
    assert len(r) == 1
    assert r[0].id == 'sq_0'

@pytest.mark.asyncio
async def test_deterministic_split_compound(decomposer):
    # LLM parroting the input query
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp({'sub_queries': [{'id': 'sq_0', 'text': 'What is RAG, and what is climate change?', 'intent_type': 'factual', 'is_dependent': False}]})):
        r = await decomposer.decompose('What is RAG, and what is climate change?')
    
    assert len(r) == 2
    assert r[0].text == 'What is RAG?'
    assert r[1].text == 'What is climate change?'

@pytest.mark.asyncio
async def test_deterministic_split_ignores_atomic(decomposer):
    # Should not split because it lacks multiple interrogative clauses
    with patch('litellm.acompletion', new_callable=AsyncMock,
               return_value=make_resp({'sub_queries': [{'id': 'sq_0', 'text': 'What are the pros and cons of RAG?', 'intent_type': 'factual', 'is_dependent': False}]})):
        r = await decomposer.decompose('What are the pros and cons of RAG?')
    
    assert len(r) == 1
    assert r[0].text == 'What are the pros and cons of RAG?'
