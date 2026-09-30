import pytest
import asyncio
from fastapi.testclient import TestClient
from api.server import app
from core.state import state_manager
from unittest.mock import patch, AsyncMock
from core import SubQuery, RetrievedChunk

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_early_retrieval_triggers_mid_stream(client):
    session_id = "sess_g2_test_1"
    state_manager.destroy(session_id)
    
    with client.websocket_connect(f"/ws/{session_id}") as websocket:
        with patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose, \
             patch('api.server.retriever.retrieve_multi', new_callable=AsyncMock) as mock_retrieve:
             
            mock_decompose.return_value = [SubQuery(id='sq_0', text='What is RAG?', intent_type='factual')]
            chunk = RetrievedChunk("doc1", "sec1", "Text", 1.0, 1)
            mock_retrieve.return_value = {'sq_0': [chunk]}
            
            # Send tokens as a single block to avoid multiple partial triggers
            websocket.send_json({"type": "token", "content": "What is RAG?"})
            
            # Read early events (blocking is fine since we mock the heavy ops and they return immediately)
            status = websocket.receive_json()
            assert status['type'] == 'status'
            assert status['action'] == 'RETRIEVE'
            
            decomp = websocket.receive_json()
            assert decomp['type'] == 'decomposed'
            
            retr = websocket.receive_json()
            assert retr['type'] == 'retrieved'
            assert retr.get('source') == 'early_retrieval'
            
            # Now send end_of_utterance
            websocket.send_json({"type": "end_of_utterance"})
            
            # Should receive them again from EOU
            status2 = websocket.receive_json()
            assert status2['type'] == 'status'
            
            decomp2 = websocket.receive_json()
            assert decomp2['type'] == 'decomposed'
            
            # Because it matched the snapshot, decomposer was NOT called again
            assert mock_decompose.call_count == 1
            assert mock_retrieve.call_count == 1
            
def test_unstable_transcripts_do_not_trigger_early(client):
    session_id = "sess_g2_test_2"
    state_manager.destroy(session_id)
    
    with client.websocket_connect(f"/ws/{session_id}") as websocket:
        with patch('api.server.decomposer.decompose', new_callable=AsyncMock) as mock_decompose:
            # Send a phrase without question markers
            websocket.send_json({"type": "token", "content": "Information"})
            websocket.send_json({"type": "token", "content": "about"})
            websocket.send_json({"type": "token", "content": "AI"})
            websocket.send_json({"type": "token", "content": "and"})
            
            import time
            time.sleep(0.5)
            assert mock_decompose.call_count == 0
