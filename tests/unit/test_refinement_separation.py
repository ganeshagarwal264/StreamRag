import pytest
from core import SubQuery, RetrievedChunk
from api.server import _handle_refinement
from core.state import state_manager, DeltaInstruction
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_requery_separation():
    # Mock WebSocket
    websocket = AsyncMock()
    
    # Setup session
    session_id = "test_sep_01"
    session = state_manager.get_or_create(session_id)
    session.tokens = "What is retrieval augmented generation? Only information after 2020.".split()
    session.pending_tokens = [] # Turn 2 just started
    
    # active_sub_queries from Turn 1
    original_sq = SubQuery(id='sq_0', text='What is retrieval augmented generation?', intent_type='factual')
    session.active_sub_queries = [original_sq]
    
    # Delta
    delta = DeltaInstruction(
        needs_update=True,
        new_constraints={'year_min': 2020},
        refinement_prompt_suffix=" Please update the answer to only include information from 2020 onward.",
        mode="REQUERY",
        chunks=None,
        augmented_query="What is retrieval augmented generation? Only information after 2020."
    )
    
    # Mock retriever
    import api.server
    api.server.retriever = AsyncMock()
    api.server.retriever.retrieve_multi.return_value = {'sq_0': [RetrievedChunk(doc_id='1', section='s', text='t', score=1, rank=1)]}
    
    # Mock synthesizer
    api.server.synthesizer = AsyncMock()
    api.server.synthesizer.synthesize_merged = AsyncMock()
    
    await _handle_refinement(websocket, session_id, delta)
    
    # 1. REQUERY retrieval receives the augmented query
    retriever_call = api.server.retriever.retrieve_multi.call_args[0][0]
    assert len(retriever_call) == 1
    assert retriever_call[0].text == "What is retrieval augmented generation? Only information after 2020."
    
    # 2. Synthesis receives the original semantic sub-query
    synth_call_sub_queries = api.server.synthesizer.synthesize_merged.call_args[0][0]
    assert len(synth_call_sub_queries) == 1
    assert synth_call_sub_queries[0].text == "What is retrieval augmented generation?"
    
    # 3. session.active_sub_queries remains the original semantic query after REQUERY
    assert session.active_sub_queries[0].text == "What is retrieval augmented generation?"
    
    # 4. The synthesis QUESTION does not contain 'Only information after 2020.'
    # (Since it uses sub_queries, it doesn't contain it)
    assert synth_call_sub_queries[0].text == "What is retrieval augmented generation?"
    
    # 5. refinement_suffix still contains "from 2020 onward"
    synth_kwargs = api.server.synthesizer.synthesize_merged.call_args[1]
    assert "from 2020 onward" in synth_kwargs['refinement_suffix']
    
    # 6. Retrieval query may contain the constraint
    assert "after 2020" in retriever_call[0].text
    
    # 7. User transcript remains unchanged
    assert session.full_transcript() == "What is retrieval augmented generation? Only information after 2020."

@pytest.mark.asyncio
async def test_filter_behavior():
    # Setup session
    session_id = "test_sep_02"
    session = state_manager.get_or_create(session_id)
    original_sq = SubQuery(id='sq_0', text='What is retrieval augmented generation?', intent_type='factual')
    session.active_sub_queries = [original_sq]
    
    # Delta FILTER
    delta = DeltaInstruction(
        needs_update=True,
        new_constraints={'year_min': 2020},
        refinement_prompt_suffix=" Please update the answer to only include information from 2020 onward.",
        mode="FILTER",
        chunks=[RetrievedChunk(doc_id='1', section='s', text='t', score=1, rank=1)],
        augmented_query=None
    )
    
    websocket = AsyncMock()
    import api.server
    api.server.synthesizer = AsyncMock()
    api.server.synthesizer.synthesize_merged = AsyncMock()
    
    await _handle_refinement(websocket, session_id, delta)
    
    # 9. Existing FILTER behavior remains unchanged
    assert session.active_sub_queries[0].text == "What is retrieval augmented generation?"
    synth_call = api.server.synthesizer.synthesize_merged.call_args[0][0]
    assert synth_call[0].text == "What is retrieval augmented generation?"
