"""
tests/unit/test_server_responsive.py

Verifies:
  1. Zero-chunk short-circuit: retriever returns 0 chunks → server emits
     insufficient_evidence + done WITHOUT attempting synthesis.
  2. Task cancellation: WebSocket remains responsive and emits reset_ack
     even while a synthesis task is in-flight.

Infrastructure strategy
-----------------------
Four components make live network/model calls and must ALL be mocked:

  a) HybridRetriever.__init__ (loaded by lifespan) → loads HF models + ChromaDB
  b) IntentController._llm_stability_check (called when heuristic confidence < threshold)
  c) IntentDecomposer.decompose (calls LiteLLM / Ollama)
  d) AnswerSynthesizer.synthesize_merged / stream_answer (calls LiteLLM / Ollama)

We mock (a) via patch.object on HybridRetriever in the server module, so the
lifespan never touches HF code.  We mock (b)-(d) via patch.object on the module-level
singleton instances that server.py creates at import time.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from core import StabilityDecision, SubQuery


# ---------------------------------------------------------------------------
# Retriever stub factories
# ---------------------------------------------------------------------------

def _make_empty_retriever():
    """0 chunks → zero-chunk short-circuit must fire."""
    mock = MagicMock()
    mock.retrieve_multi = AsyncMock(return_value={})
    mock.index_corpus = MagicMock()
    mock.get_corpus_stats = MagicMock(return_value={
        "total_chunks": 0, "collection_name": "test",
        "bm25_vocab_size": 0, "corpus_docs": 0,
    })
    return mock


def _make_chunk_retriever():
    """1 chunk → retrieval succeeds; synthesis task is spawned."""
    from core import RetrievedChunk
    chunk = RetrievedChunk(
        doc_id="ai_003", section="RAG Overview",
        text="RAG combines retrieval and generation.",
        score=1.0, rank=1, metadata={"year": 2021, "domain": "AI"},
    )
    mock = MagicMock()
    mock.retrieve_multi = AsyncMock(return_value={"sq_0": [chunk]})
    mock.index_corpus = MagicMock()
    mock.get_corpus_stats = MagicMock(return_value={
        "total_chunks": 1, "collection_name": "test",
        "bm25_vocab_size": 10, "corpus_docs": 1,
    })
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_server_zero_chunk_short_circuit():
    """
    Arrange: retriever always returns 0 chunks.
    Act:     send a token + end_of_utterance.
    Assert:  server emits insufficient_evidence and then done (no hang, no crash).
    """
    import api.server as server_module
    from fastapi.testclient import TestClient

    retriever_instance = _make_empty_retriever()

    # RETRIEVE decision so the pipeline reaches retrieval without an LLM call
    retrieve_decision = StabilityDecision(
        action="RETRIEVE", confidence=0.9, reasoning="mocked"
    )
    # Single sub-query so decompose never calls the LLM
    single_sq = [SubQuery(id="sq_0", text="Only information after 2050",
                          intent_type="factual")]

    with patch.object(server_module, "HybridRetriever",
                      return_value=retriever_instance), \
         patch.object(server_module.controller, "assess",
                      new_callable=AsyncMock,
                      return_value=retrieve_decision), \
         patch.object(server_module.decomposer, "decompose",
                      new_callable=AsyncMock,
                      return_value=single_sq):

        with TestClient(server_module.app) as client:
            server_module.retriever = retriever_instance

            with client.websocket_connect("/ws/test_zero_sc") as websocket:
                websocket.send_json({
                    "type": "token",
                    "content": "Only information after 2050",
                })
                websocket.send_json({"type": "end_of_utterance"})

                insufficient_seen = False
                done_seen = False

                for _ in range(20):
                    msg = websocket.receive_json()
                    if msg.get("type") == "error":
                        break
                    if msg.get("type") == "insufficient_evidence":
                        insufficient_seen = True
                    if msg.get("type") == "done":
                        done_seen = True
                        break

    assert insufficient_seen, \
        "Server must emit insufficient_evidence when retriever returns 0 chunks"
    assert done_seen, \
        "Server must emit done when retriever returns 0 chunks"


def test_server_synthesis_task_cancellation():
    """
    Arrange: retriever returns 1 chunk so synthesis task is spawned.
    Act:     send token + EOU, wait for retrieved event, then reset.
    Assert:  WebSocket remains responsive and emits reset_ack promptly.
    """
    import api.server as server_module
    from fastapi.testclient import TestClient

    retriever_instance = _make_chunk_retriever()

    retrieve_decision = StabilityDecision(
        action="RETRIEVE", confidence=0.9, reasoning="mocked"
    )
    single_sq = [SubQuery(id="sq_0", text="What is RAG",
                          intent_type="factual")]

    # synthesize_merged is mocked to never resolve (simulates a stalled LLM)
    # so the synthesis task stays alive until reset cancels it.
    async def _stalled_synth(*args, **kwargs):
        import asyncio
        await asyncio.sleep(60)  # will be cancelled by reset

    with patch.object(server_module, "HybridRetriever",
                      return_value=retriever_instance), \
         patch.object(server_module.controller, "assess",
                      new_callable=AsyncMock,
                      return_value=retrieve_decision), \
         patch.object(server_module.decomposer, "decompose",
                      new_callable=AsyncMock,
                      return_value=single_sq), \
         patch.object(server_module.synthesizer, "synthesize_merged",
                      side_effect=_stalled_synth):

        with TestClient(server_module.app) as client:
            server_module.retriever = retriever_instance

            with client.websocket_connect("/ws/test_cancel_sc") as websocket:
                websocket.send_json({"type": "token", "content": "What is RAG "})
                websocket.send_json({"type": "end_of_utterance"})

                # Drain until we see the retrieved event (synthesis task running)
                for _ in range(20):
                    msg = websocket.receive_json()
                    if msg.get("type") == "error":
                        break
                    if (msg.get("type") == "retrieved"
                            and msg.get("source") != "early_retrieval"):
                        break

                # Send reset while synthesis is stalled
                websocket.send_json({"type": "reset"})

                for _ in range(20):
                    msg = websocket.receive_json()
                    if msg.get("type") == "reset_ack":
                        return  # passed

    pytest.fail("WebSocket MUST remain responsive and emit reset_ack")
