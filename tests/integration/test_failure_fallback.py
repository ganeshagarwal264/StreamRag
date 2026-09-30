"""
tests/integration/test_failure_fallback.py
Verifies synthesizer and decomposer degrade gracefully on LiteLLM failure.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from core.synthesizer import AnswerSynthesizer
from core.decomposer  import IntentDecomposer
from core import SubQuery, RetrievedChunk, SynthesisResult

SAMPLE_CHUNKS = [
    RetrievedChunk(doc_id="ai_001", section="Intro",
                   text="Transformers use self-attention.", score=0.9, rank=1, metadata={}),
]

# ── Synthesizer tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesizer_emits_error_token_on_failure():
    """All retries exhausted -> user receives error token, on_done fires."""
    synth = AnswerSynthesizer()
    tokens, done_results = [], []

    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    import litellm
    with patch.object(litellm, "acompletion", new_callable=AsyncMock,
                      side_effect=Exception("API down")):
        await synth.stream_answer(
            "What are transformers?", SAMPLE_CHUNKS,
            on_token, on_citation, on_done,
        )

    assert len(done_results) == 1
    assert done_results[0].is_insufficient is True
    assert any("unavailable" in t.lower() or "error" in t.lower() for t in tokens)

@pytest.mark.asyncio
async def test_synthesizer_retries_once_then_succeeds():
    """First call fails; second succeeds -> user gets the answer."""
    synth = AnswerSynthesizer()
    tokens, done_results, call_log = [], [], []

    async def on_token(t): tokens.append(t)
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    class FakeChunk:
        choices = [type("C", (), {"delta": type("D", (), {"content": "Attention is all you need."})()})()]
        usage = None

    async def fake_acompletion(**kwargs):
        call_log.append(1)
        if len(call_log) == 1:
            raise Exception("First call fails")
        async def _gen():
            yield FakeChunk()
        return _gen()

    import litellm
    with patch.object(litellm, "acompletion", side_effect=fake_acompletion):
        await synth.stream_answer(
            "Explain attention", SAMPLE_CHUNKS,
            on_token, on_citation, on_done,
        )

    assert len(call_log) == 2, "Should retry exactly once"
    assert len(done_results) == 1

# ── Decomposer tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decomposer_falls_back_to_single_intent_on_api_failure():
    """
    When decomposer LLM fails, must return exactly one SubQuery
    (the original query as-is) rather than raising or returning [].
    """
    dec = IntentDecomposer()
    import litellm
    with patch.object(litellm, "acompletion", new_callable=AsyncMock,
                      side_effect=Exception("API error")):
        result = await dec.decompose("How do transformers work and what is RAG?")

    assert len(result) == 2
    assert result[0].id == "sq_0"
    assert result[1].id == "sq_1"
    assert "transformers" in result[0].text.lower()
    assert "rag" in result[1].text.lower()

@pytest.mark.asyncio
async def test_decomposer_timeout_triggers_fallback():
    """Timeout is treated the same as an API error."""
    import asyncio as aio
    dec = IntentDecomposer()
    import litellm
    with patch.object(litellm, "acompletion", new_callable=AsyncMock,
                      side_effect=aio.TimeoutError()):
        result = await dec.decompose("Compare climate tipping points and sea level rise")

    assert len(result) == 1
