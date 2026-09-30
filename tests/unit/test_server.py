import pytest
import json
from fastapi.testclient import TestClient
from api.server import app, controller, decomposer, retriever, synthesizer
from core.state import state_manager
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_session_refinement_lifecycle(client):
    session_id = "sess_test_refinement"
    state_manager.destroy(session_id)
    
    # A. First normal query creates turn 1
    with client.websocket_connect(f"/ws/{session_id}") as websocket:
        # We need to mock the singletons inside api.server since they are created at load
        with patch('api.server.controller.assess', new_callable=AsyncMock) as mock_assess, \
             patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
             patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve, \
             patch('api.server.synthesizer.synthesize_merged', new_callable=AsyncMock) as mock_synth:
             
            # Setup mocks for turn 1
            from core import StabilityDecision, SubQuery, RetrievedChunk, SynthesisResult
            mock_assess.return_value = StabilityDecision(action='RETRIEVE', confidence=0.9, reasoning='')
            mock_decompose.return_value = [SubQuery(id='sq_0', text='What is RAG?', intent_type='factual')]
            chunk = RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)
            mock_retrieve.return_value = {'sq_0': [chunk]}
            
            async def fake_synth(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=''):
                await on_token("Answer.")
                await on_citation("doc1", "sec1")
                res = SynthesisResult(answer="Answer.", citations=["[doc1 §sec1]"], is_insufficient=False, latency_ms=10)
                await on_done(res, 0.0)
            mock_synth.side_effect = fake_synth
            
            # Send Turn 1
            websocket.send_json({"type": "token", "content": "What is RAG?"})
            websocket.send_json({"type": "end_of_utterance"})
            

            # Read responses
            # G2 Early retrieval might push status/decomposed/retrieved first.
            # EOU pushes them again. We just read until we hit 'token'.
            event = websocket.receive_json()
            while event["type"] in ("status", "decomposed", "retrieved"):
                event = websocket.receive_json()
                
            assert event["type"] == "token"
            assert websocket.receive_json()["type"] == "citation"
            assert websocket.receive_json()["type"] == "done"
            
            session = state_manager.get_or_create(session_id)
            assert session.turn_count == 1
            assert session.current_answer == "Answer."
            assert len(session.pending_tokens) == 0
            
            # B. Second utterance: "Only information after 2020."
            with patch.object(state_manager, '_llm_extract_constraint', new_callable=AsyncMock) as mock_llm_fallback:
                websocket.send_json({"type": "token", "content": "Only "})
                websocket.send_json({"type": "token", "content": "information "})
                websocket.send_json({"type": "token", "content": "after "})
                websocket.send_json({"type": "token", "content": "2020."})
                
                # E. No LLM constraint fallback is called once per token
                mock_llm_fallback.assert_not_called()
                
                websocket.send_json({"type": "end_of_utterance"})
                
                # H. Refinement emits delta_detected
                delta_event = websocket.receive_json()
                assert delta_event["type"] == "delta_detected"
                
                # G. year_min = 2020 is detected
                assert delta_event["constraints"] == {"year_min": 2020}
                
                # C. The detector received the COMPLETE phrase
                mock_llm_fallback.assert_not_called()
                
                # Check refinement flow
                refining_event = websocket.receive_json()
                assert refining_event["type"] == "refining"
                
                # I. Refinement follows the existing FILTER/REQUERY decision
                assert delta_event["delta_mode"] in ["FILTER", "REQUERY"]
                
                retrieved_event = websocket.receive_json()
                assert retrieved_event["type"] == "retrieved"
                
                # token -> citation -> done
                assert websocket.receive_json()["type"] == "token"
                assert websocket.receive_json()["type"] == "citation"
                assert websocket.receive_json()["type"] == "done"
                
                # J. Session state remains intact after refinement
                session = state_manager.get_or_create(session_id)
                assert session.constraints == {"year_min": 2020}
                
                # K. pending_tokens are cleared
                assert len(session.pending_tokens) == 0
                
                # L. Turn count becomes 2
                assert session.turn_count == 2
