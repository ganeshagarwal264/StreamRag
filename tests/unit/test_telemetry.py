import pytest
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from api.server import app
from core.state import state_manager
from telemetry.logger import LOG_DIR, LOG_FILE

EXPECTED_EVENTS = {
    "session_start",
    "token_ingestion",
    "stability_decision",
    "decomposed",
    "retrieval",
    "delta_update",
    "refining",
    "token",
    "citation",
    "insufficient_evidence",
    "conflicting_evidence",
    "synthesis",
    "error",
    "session_reset",
    "session_end",
}

def get_logged_events():
    log_path = Path(LOG_DIR) / LOG_FILE
    if not log_path.exists():
        return set()
    events = set()
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "event_type" in data:
                        events.add(data["event_type"])
                except Exception:
                    pass
    return events

@pytest.fixture(autouse=True)
def clear_logs():
    log_path = Path(LOG_DIR) / LOG_FILE
    if log_path.exists():
        log_path.unlink()
    yield
    # Cleanup after test

def test_telemetry_coverage():
    from core import StabilityDecision, SubQuery, RetrievedChunk, SynthesisResult
    
    # We will trigger the various paths by mocking the components slightly 
    # where needed so we don't have to wait for real LLMs or real retrieval.
    
    with TestClient(app) as client:
        # 1. Normal session (session_start, token_ingestion, stability_decision, decomposed, retrieval, token, citation, synthesis, session_end)
        session_id = "sess_telem_1"
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess, \
                 patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
                 patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve, \
                 patch('api.server.synthesizer.synthesize_merged', new_callable=AsyncMock) as mock_synth, \
                 patch('api.server.synthesizer._check_contradictions', new_callable=AsyncMock) as mock_contra:
                
                mock_assess.return_value = StabilityDecision(action='RETRIEVE', confidence=0.9, reasoning='')
                mock_decompose.return_value = [SubQuery(id='sq_0', text='What is RAG?', intent_type='factual')]
                chunk = RetrievedChunk("ai_001", "sec1", "Text", 1.0, 1)
                mock_retrieve.return_value = {'sq_0': [chunk]}
                mock_contra.return_value = []
                
                async def fake_synth(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=''):
                    await on_token("Answer.")
                    await on_citation("ai_001", "sec1")
                    res = SynthesisResult(answer="Answer.", citations=["[ai_001 Â§sec1]"], is_insufficient=False, latency_ms=10)
                    await on_done(res, 0.0)
                mock_synth.side_effect = fake_synth
                
                websocket.send_json({"type": "token", "content": "What"})
                websocket.send_json({"type": "token", "content": "is"})
                websocket.send_json({"type": "token", "content": "RAG?"})
                websocket.send_json({"type": "end_of_utterance"})
                
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "done":
                        break
        
        # 2. Refinement (delta_update, refining)
        session_id_2 = "sess_telem_2"
        with client.websocket_connect(f"/ws/{session_id_2}") as websocket:
            with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess, \
                 patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
                 patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve, \
                 patch('api.server.synthesizer.synthesize_merged', new_callable=AsyncMock) as mock_synth:
                 
                mock_assess.return_value = StabilityDecision(action='RETRIEVE', confidence=0.9, reasoning='')
                mock_decompose.return_value = [SubQuery(id='sq_0', text='What is RAG?', intent_type='factual')]
                chunk = RetrievedChunk("ai_001", "sec1", "Text", 1.0, 1)
                mock_retrieve.return_value = {'sq_0': [chunk]}
                
                async def fake_synth_refine(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=''):
                    await on_token("Refined.")
                    await on_citation("ai_001", "sec1")
                    res = SynthesisResult(answer="Refined.", citations=["[ai_001 §sec1]"], is_insufficient=False, latency_ms=10)
                    await on_done(res, 0.0)
                mock_synth.side_effect = fake_synth_refine
                
                # Turn 1 to get an answer
                websocket.send_json({"type": "token", "content": "What"})
                websocket.send_json({"type": "token", "content": "is"})
                websocket.send_json({"type": "token", "content": "RAG?"})
                websocket.send_json({"type": "end_of_utterance"})
                
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "done":
                        break
                 
                # Turn 2: Late constraint
                websocket.send_json({"type": "token", "content": "only"})
                websocket.send_json({"type": "token", "content": "after"})
                websocket.send_json({"type": "token", "content": "2020"})
                websocket.send_json({"type": "end_of_utterance"})
                
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "done":
                        break
            
            # 3. session_reset
            websocket.send_json({"type": "reset"})
            websocket.receive_json() # reset_ack
            
        # 4. Insufficient evidence
        session_id_3 = "sess_telem_3"
        with client.websocket_connect(f"/ws/{session_id_3}") as websocket:
            with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess, \
                 patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
                 patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve, \
                 patch('api.server.synthesizer.synthesize_merged', new_callable=AsyncMock) as mock_synth:
                 
                mock_assess.return_value = StabilityDecision(action='RETRIEVE', confidence=0.9, reasoning='')
                mock_decompose.return_value = [SubQuery(id='sq_0', text='Explain topological quantum computing', intent_type='factual')]
                mock_retrieve.return_value = {'sq_0': []}
                
                async def fake_synth_insuf(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=''):
                    res = SynthesisResult(answer="INSUFFICIENT_EVIDENCE", citations=[], is_insufficient=True, latency_ms=10)
                    await on_done(res, 0.0)
                mock_synth.side_effect = fake_synth_insuf
                
                websocket.send_json({"type": "token", "content": "Explain topological quantum computing"})
                websocket.send_json({"type": "end_of_utterance"})
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "done":
                        break
                        
        # 5. Conflicting evidence
        session_id_4 = "sess_telem_4"
        with client.websocket_connect(f"/ws/{session_id_4}") as websocket:
            with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess, \
                 patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
                 patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve, \
                 patch('api.server.synthesizer.synthesize_merged', new_callable=AsyncMock) as mock_synth, \
                 patch('api.server.synthesizer._check_contradictions', new_callable=AsyncMock) as mock_contra:
                 
                mock_assess.return_value = StabilityDecision(action='RETRIEVE', confidence=0.9, reasoning='')
                # compound query
                mock_decompose.return_value = [
                    SubQuery(id='sq_0', text='Q1', intent_type='factual'),
                    SubQuery(id='sq_1', text='Q2', intent_type='factual')
                ]
                chunk_a = RetrievedChunk("ai_001", "sec1", "Text A", 1.0, 1)
                chunk_b = RetrievedChunk("ai_002", "sec2", "Text B", 0.9, 2)
                mock_retrieve.return_value = {'sq_0': [chunk_a], 'sq_1': [chunk_b]}
                
                mock_contra.return_value = [(chunk_a, chunk_b)]
                
                async def fake_synth_contra(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=''):
                    res = SynthesisResult(answer="Answer.", citations=[], is_insufficient=False, latency_ms=10)
                    await on_done(res, 0.0)
                mock_synth.side_effect = fake_synth_contra
                
                websocket.send_json({"type": "token", "content": "Compound query"})
                websocket.send_json({"type": "end_of_utterance"})
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "done":
                        break
                        
        # 6. Error
        session_id_5 = "sess_telem_5"
        with client.websocket_connect(f"/ws/{session_id_5}") as websocket:
            with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess:
                mock_assess.side_effect = Exception("Simulated error")
                
                websocket.send_json({"type": "token", "content": "Break"})
                websocket.send_json({"type": "end_of_utterance"})
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "error":
                        break
        
    logged_events = get_logged_events()
    missing_events = EXPECTED_EVENTS - logged_events
    
    print(f"\nEXPECTED EVENTS: {EXPECTED_EVENTS}")
    print(f"LOGGED EVENTS:   {logged_events}")
    print(f"MISSING EVENTS:  {missing_events}")
    
    coverage = len(logged_events.intersection(EXPECTED_EVENTS)) / len(EXPECTED_EVENTS)
    print(f"COVERAGE: {coverage * 100:.1f}%")
    
    assert len(missing_events) == 0, f"Missing telemetry events: {missing_events}"
