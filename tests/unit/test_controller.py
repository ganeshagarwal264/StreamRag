import pytest
from unittest.mock import patch, AsyncMock
from core import SessionState, StabilityDecision
from core.controller import IntentController

@pytest.fixture
def controller():
    return IntentController()

@pytest.fixture
def session():
    return SessionState(session_id='test')

@pytest.mark.asyncio
async def test_suppress_next_slide(controller, session):
    d = await controller.assess('next slide please', session)
    assert d.action == 'SUPPRESS'

@pytest.mark.asyncio
async def test_suppress_go_back(controller, session):
    d = await controller.assess('go back to previous slide', session)
    assert d.action == 'SUPPRESS'

@pytest.mark.asyncio
async def test_wait_too_few_tokens(controller, session):
    d = await controller.assess('um what', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_trailing_and(controller, session):
    d = await controller.assess('explain neural networks and', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_trailing_comma(controller, session):
    d = await controller.assess('what is machine learning,', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_retrieve_clear_question(controller, session):
    # Mock LLM call since confidence may be borderline
    with patch.object(controller, '_llm_stability_check', new_callable=AsyncMock) as mock:
        mock.return_value = StabilityDecision('RETRIEVE', 0.9, 'clear complete question')
        d = await controller.assess('What is retrieval augmented generation and how does it reduce hallucinations', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_suppress_scroll(controller, session):
    d = await controller.assess('scroll down please', session)
    assert d.action == 'SUPPRESS'

@pytest.mark.asyncio
async def test_wait_ellipsis(controller, session):
    d = await controller.assess('what about transformers...', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_retrieve_concise_question(controller, session):
    d = await controller.assess('Who was Vaswani?', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_retrieve_short_explain(controller, session):
    d = await controller.assess('Explain RAG', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_retrieve_short_why(controller, session):
    d = await controller.assess('Why does RAG help?', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_retrieve_tell_me(controller, session):
    d = await controller.assess('Tell me about Vaswani', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_retrieve_and_what(controller, session):
    d = await controller.assess('And what about RAG?', session)
    assert d.action == 'RETRIEVE'

@pytest.mark.asyncio
async def test_wait_who_is(controller, session):
    d = await controller.assess('Who is?', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_what_is(controller, session):
    d = await controller.assess('What is?', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_why(controller, session):
    d = await controller.assess('Why?', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_what_about_the(controller, session):
    d = await controller.assess('And also what about the', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_only_after(controller, session):
    d = await controller.assess('only after 2020', session)
    assert d.action == 'WAIT'

@pytest.mark.asyncio
async def test_wait_after_2020(controller, session):
    d = await controller.assess('after 2020', session)
    assert d.action == 'WAIT'
