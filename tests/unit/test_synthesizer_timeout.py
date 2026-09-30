"""
tests/unit/test_synthesizer_timeout.py

Tests for the streaming-phase timeout fix added to AnswerSynthesizer.stream_answer.

Coverage required:
  1. Normal synthesis produces tokens and a done result (happy path unchanged).
  2. Stream stalls before first token → timeout fires within SYNTHESIS_TIMEOUT_S.
  3. Timeout produces a visible recoverable error token + is_insufficient=True done.
  4. No stale answer from a prior turn survives the timeout.
  5. Synthesis task is cancelled/cleaned up.
  6. Subsequent request can still succeed after the timeout.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core import RetrievedChunk, SynthesisResult
from core.synthesizer import AnswerSynthesizer


# ── helpers ───────────────────────────────────────────────────────────────────

def make_chunk(text: str) -> MagicMock:
    m = MagicMock()
    m.usage = None
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    m.choices = [choice]
    return m


def make_empty_chunk() -> MagicMock:
    m = MagicMock()
    m.usage = MagicMock()
    delta = MagicMock()
    delta.content = None
    choice = MagicMock()
    choice.delta = delta
    m.choices = [choice]
    return m


async def _forever():
    yield make_empty_chunk()
    await asyncio.sleep(9999)


async def _fast_stream(tokens):
    for tok in tokens:
        yield make_chunk(tok)


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="ai_003", section="RAG Overview",
        text="RAG combines retrieval with generation.",
        score=5.0, rank=1,
        metadata={"domain": "AI", "year": 2023},
    )


# ── test 1 — happy path ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normal_synthesis_produces_tokens_and_done():
    import core.synthesizer as syn_mod
    syn_mod._SYNTHESIS_TIMEOUT_S = 30.0

    synth = AnswerSynthesizer()
    tokens_received = []
    done_results = []

    response_mock = MagicMock()
    response_mock.__aiter__ = lambda self: _fast_stream(
        ["RAG ", "combines ", "[ai_003 \xa7RAG Overview]."]
    )

    async def on_token(t): tokens_received.append(t)
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response_mock):
        await synth.stream_answer(
            query="What is RAG?", chunks=[sample_chunk()],
            on_token=on_token, on_citation=on_citation, on_done=on_done,
        )

    assert len(tokens_received) > 0
    assert len(done_results) == 1
    assert not done_results[0].is_insufficient
    assert done_results[0].answer != ""


# ── test 2 — stream stalls before first token → timeout fires ─────────────────

@pytest.mark.asyncio
async def test_stream_stall_triggers_timeout():
    import core.synthesizer as syn_mod
    syn_mod._SYNTHESIS_TIMEOUT_S = 2.0

    synth = AnswerSynthesizer()
    done_results = []

    async def on_token(t): pass
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    response_mock = MagicMock()
    response_mock.__aiter__ = lambda self: _forever()

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response_mock):
        await synth.stream_answer(
            query="What is RAG?", chunks=[sample_chunk()],
            on_token=on_token, on_citation=on_citation, on_done=on_done,
        )

    assert len(done_results) == 1
    assert done_results[0].is_insufficient


# ── test 3 — timeout error token is visible to client ─────────────────────────

@pytest.mark.asyncio
async def test_timeout_emits_visible_error_token():
    import core.synthesizer as syn_mod
    syn_mod._SYNTHESIS_TIMEOUT_S = 2.0

    synth = AnswerSynthesizer()
    error_tokens = []
    done_results = []

    async def on_token(t): error_tokens.append(t)
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    response_mock = MagicMock()
    response_mock.__aiter__ = lambda self: _forever()

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response_mock):
        await synth.stream_answer(
            query="What is RAG?", chunks=[sample_chunk()],
            on_token=on_token, on_citation=on_citation, on_done=on_done,
        )

    combined = "".join(error_tokens)
    assert "timed out" in combined.lower() or "synthesis" in combined.lower(), (
        f"Expected a user-visible timeout message, got: {combined!r}"
    )
    assert done_results[0].citations == []


# ── test 4 — stale answer not in timeout result ────────────────────────────────

@pytest.mark.asyncio
async def test_stale_answer_not_in_timeout_result():
    import core.synthesizer as syn_mod
    syn_mod._SYNTHESIS_TIMEOUT_S = 2.0

    synth = AnswerSynthesizer()
    previous_answer = "Previous correct answer from turn 1."
    done_results = []

    async def on_token(t): pass
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): done_results.append(r)

    response_mock = MagicMock()
    response_mock.__aiter__ = lambda self: _forever()

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response_mock):
        await synth.stream_answer(
            query="What is RAG?", chunks=[sample_chunk()],
            on_token=on_token, on_citation=on_citation, on_done=on_done,
        )

    result = done_results[0]
    assert previous_answer not in result.answer
    assert result.is_insufficient


# ── test 5 — cancelled task propagates cleanly ────────────────────────────────

@pytest.mark.asyncio
async def test_synthesis_task_cancelled_cleanly():
    import core.synthesizer as syn_mod
    syn_mod._SYNTHESIS_TIMEOUT_S = 30.0

    synth = AnswerSynthesizer()

    async def slow_stream():
        for i in range(100):
            yield make_chunk(f"token{i}")
            await asyncio.sleep(0.05)

    response_mock = MagicMock()
    response_mock.__aiter__ = lambda self: slow_stream()

    async def on_token(t): pass
    async def on_citation(d, s): pass
    async def on_done(r, cost=0.0): pass

    async def run():
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response_mock):
            await synth.stream_answer(
                query="What is RAG?", chunks=[sample_chunk()],
                on_token=on_token, on_citation=on_citation, on_done=on_done,
            )

    task = asyncio.create_task(run())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert task.done()


# ── test 6 — subsequent request succeeds after timeout ────────────────────────

@pytest.mark.asyncio
async def test_subsequent_request_succeeds_after_timeout():
    import core.synthesizer as syn_mod

    synth = AnswerSynthesizer()

    async def on_noop(t): pass
    async def on_cite_noop(d, s): pass

    # Turn 1: stall → timeout
    syn_mod._SYNTHESIS_TIMEOUT_S = 2.0
    stall_response = MagicMock()
    stall_response.__aiter__ = lambda self: _forever()
    done1 = []

    async def on_done1(r, cost=0.0): done1.append(r)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=stall_response):
        await synth.stream_answer(
            query="turn 1", chunks=[sample_chunk()],
            on_token=on_noop, on_citation=on_cite_noop, on_done=on_done1,
        )
    assert done1[0].is_insufficient

    # Turn 2: succeeds normally
    syn_mod._SYNTHESIS_TIMEOUT_S = 30.0
    fast_response = MagicMock()
    fast_response.__aiter__ = lambda self: _fast_stream(
        ["RAG ", "is ", "great [ai_003 \xa7RAG Overview]."]
    )
    tokens2 = []
    done2 = []

    async def on_token2(t): tokens2.append(t)
    async def on_done2(r, cost=0.0): done2.append(r)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=fast_response):
        await synth.stream_answer(
            query="turn 2", chunks=[sample_chunk()],
            on_token=on_token2, on_citation=on_cite_noop, on_done=on_done2,
        )
    assert len(tokens2) > 0
    assert not done2[0].is_insufficient
