import pytest
from core import SessionState, RetrievedChunk, SubQuery, DeltaInstruction
from core.state import StateManager

@pytest.fixture
def sm():
    return StateManager()

def test_create_session(sm):
    s = sm.get_or_create('s1')
    assert s.session_id == 's1'
    assert s.tokens == []
    assert s.turn_count == 0

def test_get_existing_session(sm):
    s1 = sm.get_or_create('s2')
    s2 = sm.get_or_create('s2')
    assert s1 is s2

def test_append_token(sm):
    sm.get_or_create('s3')
    sm.append_token('s3', 'hello')
    sm.append_token('s3', 'world')
    s = sm.get_or_create('s3')
    assert s.tokens == ['hello', 'world']
    assert s.pending_tokens == ['hello', 'world']

def test_session_isolation(sm):
    sm.append_token('sa', 'alpha')
    sm.append_token('sb', 'beta')
    assert sm.get_or_create('sa').tokens == ['alpha']
    assert sm.get_or_create('sb').tokens == ['beta']

def test_destroy_session(sm):
    sm.get_or_create('sd')
    sm.destroy('sd')
    assert 'sd' not in sm._sessions

def test_mark_retrieved_clears_pending(sm):
    sm.get_or_create('sr')
    sm.append_token('sr', 'what')
    chunk = RetrievedChunk(doc_id='ai_001', section='Intro', text='test', score=0.9, rank=1)
    sq = SubQuery(id='sq_0', text='what is', intent_type='factual')
    sm.mark_retrieved('sr', [chunk], [sq])
    s = sm.get_or_create('sr')
    assert s.pending_tokens == []
    assert s.turn_count == 1
    assert len(s.retrieved_chunks) == 1

@pytest.mark.asyncio
async def test_detect_year_after(sm):
    sm.get_or_create('sy1')
    delta = await sm.detect_late_constraints('sy1', ['only', 'after', '2022'])
    assert delta.needs_update == True
    assert delta.new_constraints.get('year_min') == 2022

@pytest.mark.asyncio
async def test_detect_year_since(sm):
    sm.get_or_create('sy2')
    delta = await sm.detect_late_constraints('sy2', ['since', '2018'])
    assert delta.needs_update == True
    assert 'year_min' in delta.new_constraints

@pytest.mark.asyncio
async def test_no_duplicate_constraint(sm):
    sm.get_or_create('sdup')
    delta1 = await sm.detect_late_constraints('sdup', ['after', '2020'])
    assert delta1.needs_update == True
    delta2 = await sm.detect_late_constraints('sdup', ['after', '2020'])
    assert delta2.needs_update == False

@pytest.mark.asyncio
async def test_open_vocab_constraint_uses_llm_fallback():
    """When regex finds nothing, LLM fallback extracts a domain constraint."""
    import json
    from unittest.mock import patch, AsyncMock
    sm = StateManager()
    sm.get_or_create("sv1")

    fake_resp = type("R", (), {
        "choices": [type("C", (), {
            "message": type("M", (), {
                "content": json.dumps(
                    {"field": "domain", "op": "eq", "value": "Space"}
                )
            })()
        })()]
    })()

    import litellm
    with patch.object(litellm, "acompletion", new_callable=AsyncMock, return_value=fake_resp):
        delta = await sm.detect_late_constraints("sv1", ["only", "space", "missions"])

    assert delta.needs_update is True
    assert delta.new_constraints.get("domain") == "Space"

def test_citation_key_format():
    chunk = RetrievedChunk(doc_id='ai_001', section='Introduction', text='', score=0.9, rank=1)
    assert chunk.citation_key == '[ai_001 §Introduction]'

def test_update_answer(sm):
    sm.get_or_create('sa2')
    sm.update_answer('sa2', 'Test answer', ['[ai_001 §Intro]'])
    s = sm.get_or_create('sa2')
    assert s.current_answer == 'Test answer'
