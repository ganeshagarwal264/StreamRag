# api/server.py
"""
FastAPI WebSocket server for Streaming Live RAG.

Event protocol (JSON):
  Client -> Server:
    {"type": "token", "content": "word"}           # stream speech tokens
    {"type": "end_of_utterance"}                    # trigger stability check + retrieve
    {"type": "reset"}                               # clear session
  
  Server -> Client:
    {"type": "status", "action": "WAIT|RETRIEVE|SUPPRESS", "confidence": 0.9, "reasoning": "..."}  
    {"type": "decomposed", "sub_queries": [...]}     # after decomposition
    {"type": "retrieved", "chunk_count": 5, "latency_ms": 120}  # retrieval done
    {"type": "token", "content": "..."}              # LLM token stream
    {"type": "citation", "doc_id": "ai_001", "section": "..."}  # citation found
    {"type": "done", "latency_ms": 420, "citation_count": 3}    # synthesis done
    {"type": "insufficient_evidence", "clarifying_question": "..."}  # no evidence
    {"type": "delta_detected", "constraints": {...}}  # late constraint found
    {"type": "refining", "reason": "late_constraint"}  # re-synthesizing
    {"type": "reset_ack"}                           # session cleared
    {"type": "error", "message": "..."}             # server-side error
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio, json, os, time, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from core import SubQuery, RetrievedChunk
from core.state import state_manager
from core.controller import IntentController
from core.decomposer import IntentDecomposer
from core.retriever import HybridRetriever
from core.synthesizer import AnswerSynthesizer
from telemetry.logger import logger

# Module-level singletons (lightweight, init at import)
controller = IntentController()
decomposer = IntentDecomposer()
synthesizer = AnswerSynthesizer()
retriever: HybridRetriever = None  # heavy init at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.log_session_event('system', 'startup', {'model': os.environ.get('LLM_MODEL', 'gemini/gemini-3.6-flash')})
    retriever = HybridRetriever()
    corpus_path = Path(__file__).parent.parent / 'data' / 'corpus.json'
    if corpus_path.exists():
        with open(corpus_path) as f:
            docs = json.load(f)
        retriever.index_corpus(docs)
        logger.log_session_event('system', 'corpus_loaded', {'chunk_count': len(docs)})
    else:
        logger.log_session_event('system', 'corpus_missing', {'path': str(corpus_path)})
    yield
    # Cleanup — nothing needed since state is ephemeral

synthesis_tasks = {}

app = FastAPI(title='Streaming Live RAG', version='1.0.0', lifespan=lifespan)

@app.get('/')
async def serve_ui():
    ui_path = Path(__file__).parent.parent / 'ui' / 'index.html'
    return HTMLResponse(content=ui_path.read_text(encoding='utf-8'))

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'model': os.environ.get('LLM_MODEL', 'gemini/gemini-3.6-flash'),
        'active_sessions': len(state_manager.list_sessions()),
        'corpus_stats': retriever.get_corpus_stats() if retriever else {}
    }

@app.get('/corpus/stats')
async def corpus_stats():
    if not retriever:
        return {'error': 'retriever not initialized'}
    return retriever.get_corpus_stats()

@app.websocket('/ws/{session_id}')
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.log_session_event(session_id, 'session_start')
    

    # ── G2 EARLY RETRIEVAL STATE ──
    speculative_task = None
    speculative_snapshot = ""
    speculative_result = None

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get('type', '')
            
            if event_type == 'token':
                token = data.get('content', '').strip()
                if not token:
                    continue
                state_manager.append_token(session_id, token)
                session = state_manager.get_or_create(session_id)
                logger.log_token_ingestion(session_id, session.turn_count, 1)

                # ── G2 EARLY RETRIEVAL ──
                if not session.current_answer and retriever is not None:
                    partial = session.pending_transcript()
                    if partial != speculative_snapshot and len(partial.split()) >= 2:
                        if speculative_task and not speculative_task.done():
                            speculative_task.cancel()
                        
                        # Quick heuristic assessment
                        _ct = [t for t in partial.split() if t.lower() not in controller.FILLER_WORDS]
                        _last = "".join(c for c in (partial.split()[-1].lower() if partial.split() else "") if c.isalnum())
                        _has_q = any(t.lower() in controller.QUESTION_MARKERS for t in _ct)
                        _dang = _last in controller.DANGLING_CONJUNCTIONS
                        _inc = partial.strip().endswith(',') or partial.strip().endswith('...')
                        _nav = any(re.search(r'' + re.escape(p) + r'', partial.lower()) for p in controller.NAVIGATIONAL_PHRASES)

                        is_concise = _has_q and len(_ct) >= 2
                        has_enough = len(_ct) >= int(os.environ.get("MIN_CONTENT_TOKENS", "5"))
                        eligible = is_concise or has_enough

                        if not _nav and not _dang and not _inc and eligible and _has_q:
                            speculative_snapshot = partial
                            speculative_result = None

                            async def _speculative_retrieve(snap: str, sess_id: str):
                                nonlocal speculative_result
                                try:
                                    # Send status to UI to show we started retrieval
                                    try:
                                        await websocket.send_json({'type': 'status', 'action': 'RETRIEVE', 'confidence': 0.9, 'reasoning': 'early speculative hit'})
                                    except Exception: pass
                                    
                                    sqs = await decomposer.decompose(snap)
                                    cbq = await retriever.retrieve_multi(sqs)
                                    speculative_result = {'chunks_by_query': cbq, 'sub_queries': sqs, 'snapshot': snap}
                                    logger.log_session_event(sess_id, 'early_retrieval_done', {
                                        'snapshot_len': len(snap),
                                        'chunk_count': sum(len(v) for v in cbq.values())
                                    })
                                    
                                    # Update UI
                                    try:
                                        await websocket.send_json({
                                            'type': 'decomposed',
                                            'sub_queries': [{'id': sq.id, 'text': sq.text, 'intent_type': sq.intent_type} for sq in sqs],
                                            'fallback': False
                                        })
                                        total_c = sum(len(v) for v in cbq.values())
                                        await websocket.send_json({'type': 'retrieved', 'chunk_count': total_c, 'latency_ms': 100.0, 'source': 'early_retrieval'})
                                    except Exception: pass
                                except asyncio.CancelledError:
                                    pass
                                except Exception as exc:
                                    pass

                            speculative_task = asyncio.create_task(_speculative_retrieve(partial, session_id))

            elif event_type == 'end_of_utterance':
                session = state_manager.get_or_create(session_id)
                transcript = session.pending_transcript()
                if not transcript.strip():
                    continue
                
                # Check for late-arriving constraints AT END OF UTTERANCE
                if session.current_answer:
                    delta = await state_manager.detect_late_constraints(session_id, session.pending_tokens)
                    if delta.needs_update:
                        logger.log_delta_update(session_id, delta)
                        await websocket.send_json({
                            'type': 'delta_detected', 
                            'constraints': session.constraints,
                            'delta_mode': delta.mode
                        })
                        
                        if session_id in synthesis_tasks:
                            synthesis_tasks[session_id].cancel()
                        
                        task = asyncio.create_task(_handle_refinement(websocket, session_id, delta))
                        synthesis_tasks[session_id] = task
                        continue
                
                # Step 1: Stability check
                try:
                    decision = await controller.assess(transcript, session)
                except Exception as e:
                    logger.log_error(session_id, session.turn_count, f'Stability check failed: {e}', True, 'StabilityError')
                    await websocket.send_json({'type': 'error', 'message': f'Stability check failed: {e}'})
                    continue
                
                logger.log_stability_decision(session_id, transcript, decision)
                await websocket.send_json({
                    'type': 'status',
                    'action': decision.action,
                    'confidence': round(decision.confidence, 3),
                    'reasoning': decision.reasoning
                })
                

                if decision.action != 'RETRIEVE':
                    continue
                
                # New standalone semantic question -> Clear previous constraints
                if session.constraints:
                    session.constraints.clear()
                    try:
                        await websocket.send_json({'type': 'constraints_cleared'})
                    except Exception: pass
                
                # Step 2: Decompose
                sub_queries = None
                chunks_by_query = None
                
                # Check if we can reuse speculative result
                if speculative_result and speculative_result.get('snapshot') == transcript:
                    sub_queries = speculative_result['sub_queries']
                    chunks_by_query = speculative_result['chunks_by_query']
                    logger.log_session_event(session_id, 'early_retrieval_used', {'snapshot_len': len(transcript)})
                else:
                    if speculative_task and not speculative_task.done():
                        speculative_task.cancel()
                    
                    try:
                        sub_queries = await decomposer.decompose(transcript)
                    except Exception as e:
                        import logging
                        logging.warning(f"Server-level decomposition fallback triggered: {e}")
                        sub_queries = decomposer.deterministic_split(transcript)
                        if len(sub_queries) < 2:
                            sub_queries = [SubQuery(id='sq_0', text=transcript, intent_type='factual')]
                    
                    if len(sub_queries) == 1:
                        sq_text = sub_queries[0].text.strip()
                        import string
                        def normalize(s):
                            return s.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")
                        if normalize(sq_text) == normalize(transcript):
                            fallback_sqs = decomposer.deterministic_split(transcript)
                            if len(fallback_sqs) > 1:
                                sub_queries = fallback_sqs
                                
                decomp_fallback = len(sub_queries) == 1 and len(transcript.split()) > 8
                logger.log_decomposed(session_id, session.turn_count, len(sub_queries), decomp_fallback)
                await websocket.send_json({
                    'type': 'decomposed',
                    'sub_queries': [{'id': sq.id, 'text': sq.text, 'intent_type': sq.intent_type} for sq in sub_queries],
                    'fallback': decomp_fallback,
                })
                
                # Step 3 & 4: Retrieve and Synthesize
                if session_id in synthesis_tasks:
                    synthesis_tasks[session_id].cancel()
                    
                async def retrieve_and_synthesize(sess_id, trans, sqs, cbq_pre):
                    t0 = time.time()
                    cbq = cbq_pre
                    if not cbq:
                        try:
                            cbq = await retriever.retrieve_multi(sqs)
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            logger.log_error(sess_id, session.turn_count, f'Retrieval failed: {e}', True, 'RetrievalError')
                            try:
                                await websocket.send_json({'type': 'error', 'message': f'Retrieval failed: {e}'})
                            except Exception: pass
                            return
                        
                    retrieval_ms = (time.time() - t0) * 1000
                    if not cbq_pre:
                        logger.log_retrieval(sess_id, sqs, cbq, retrieval_ms)
                    
                    total_chunks = sum(len(v) for v in cbq.values())
                    try:
                        await websocket.send_json({'type': 'retrieved', 'chunk_count': total_chunks, 'latency_ms': round(retrieval_ms, 1)})
                    except Exception: pass
                    
                    all_chunks = [c for cs in cbq.values() for c in cs]
                    state_manager.mark_retrieved(sess_id, all_chunks, sqs)
                    
                    if len(all_chunks) == 0:
                        try:
                            msg = "I don't have enough evidence in the supplied corpus to answer this."
                            state_manager.update_answer(sess_id, msg, [])
                            await websocket.send_json({'type': 'token', 'content': msg})
                            await websocket.send_json({'type': 'insufficient_evidence', 'clarifying_question': ''})
                            await websocket.send_json({'type': 'done', 'latency_ms': 0.0, 'citation_count': 0})
                        except Exception: pass
                        logger.log_insufficient_evidence(sess_id, session.turn_count)
                        return
                        
                    try:
                        await _handle_synthesis(websocket, sess_id, trans, sqs, cbq)
                    except asyncio.CancelledError:
                        pass

                task = asyncio.create_task(retrieve_and_synthesize(session_id, transcript, sub_queries, chunks_by_query))
                synthesis_tasks[session_id] = task

            elif event_type == 'reset':
                if session_id in synthesis_tasks:
                    synthesis_tasks[session_id].cancel()
                    del synthesis_tasks[session_id]
                state_manager.destroy(session_id)
                await websocket.send_json({'type': 'reset_ack'})
                logger.log_session_event(session_id, 'session_reset')
    
    except WebSocketDisconnect:
        if session_id in synthesis_tasks:
            synthesis_tasks[session_id].cancel()
            del synthesis_tasks[session_id]
        state_manager.destroy(session_id)
        logger.log_session_event(session_id, 'session_end')
    except Exception as e:
        session = state_manager.get_or_create(session_id)
        logger.log_error(session_id, session.turn_count, str(e), True, 'GlobalError')
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e),
                'recoverable': True,
            })
        except Exception:
            pass

async def _handle_synthesis(websocket, session_id, transcript, sub_queries, chunks_by_query, refinement_suffix: str = ''):
    session = state_manager.get_or_create(session_id)
    token_index = [0]
    async def on_token(token):
        token_index[0] += 1
        logger.log_token(session_id, session.turn_count, token_index[0])
        await websocket.send_json({'type': 'token', 'content': token})
    async def on_citation(doc_id, section): 
        logger.log_citation(session_id, session.turn_count, doc_id, section)
        await websocket.send_json({'type': 'citation', 'doc_id': doc_id, 'section': section})
    async def on_done(result, cost_usd=0.0):
        state_manager.update_answer(session_id, result.answer, result.citations)
        logger.log_synthesis(session_id, transcript, result, cost_usd)
        if result.is_insufficient:
            logger.log_insufficient_evidence(session_id, session.turn_count)
            await websocket.send_json({
                'type': 'insufficient_evidence',
                'clarifying_question': result.clarifying_question or ''
            })
        await websocket.send_json({
            'type': 'done',
            'latency_ms': round(result.latency_ms, 1),
            'citation_count': len(result.citations)
        })
    contradictions = []
    if len(sub_queries) > 1:
        contradictions = await synthesizer._check_contradictions(chunks_by_query, sub_queries)
    if contradictions:
        doc_ids = [f"{a.doc_id} §{a.section}" for a, b in contradictions] + [f"{b.doc_id} §{b.section}" for a, b in contradictions]
        logger.log_conflicting_evidence(session_id, session.turn_count, list(set(doc_ids)))
        await websocket.send_json({
            'type': 'conflicting_evidence',
            'pairs': [
                {"chunk_a": f"{a.doc_id} §{a.section}", "chunk_b": f"{b.doc_id} §{b.section}"}
                for a, b in contradictions
            ],
        })
        refinement_suffix += (
            " Where retrieved sources conflict on facts, present BOTH claims "
            "with their full citations and explicitly note the discrepancy."
        )

    try:
        await synthesizer.synthesize_merged(sub_queries, chunks_by_query, on_token, on_citation, on_done, refinement_suffix=refinement_suffix)
    except (asyncio.CancelledError, RuntimeError): pass

async def _handle_refinement(websocket, session_id, delta: 'DeltaInstruction'):
    """
    Handle a late-constraint refinement cycle.

    FILTER mode (cheap path):
        delta.chunks already holds the filtered candidate set — go straight
        to synthesis without any retrieval.

    REQUERY mode (starvation guard):
        The filtered candidate set was too small (< MIN_VIABLE_CHUNKS).
        Run a fresh hybrid retrieval scoped to delta.augmented_query, merge
        the new results into state, then synthesise.
    """
    session = state_manager.get_or_create(session_id)
    synthesis_sub_queries = list(session.active_sub_queries or [
        SubQuery(id='sq_0', text=session.pending_transcript() or session.full_transcript(), intent_type='factual')
    ])

    logger.log_refining(session_id, session.turn_count, delta.mode)
    await websocket.send_json({
        'type': 'refining',
        'reason': 'late_constraint',
        'delta_mode': delta.mode,
    })

    if delta.mode == 'FILTER':
        # Cheap path: use pre-filtered chunks directly
        chunks_by_query = {'sq_0': delta.chunks or []}
        
        if len(delta.chunks or []) == 0:
            msg = "I don't have enough evidence in the supplied corpus to answer this."
            state_manager.update_answer(session_id, msg, [])
            
            # FIXED: Mark retrieved to clear pending_tokens!
            state_manager.mark_retrieved(session_id, [], synthesis_sub_queries)
            try:
                await websocket.send_json({'type': 'token', 'content': msg})
                await websocket.send_json({'type': 'insufficient_evidence', 'clarifying_question': ''})
                await websocket.send_json({'type': 'done', 'latency_ms': 0.0, 'citation_count': 0})
            except Exception: pass
            logger.log_insufficient_evidence(session_id, session.turn_count)
            return

        all_chunks = delta.chunks or []
        state_manager.mark_retrieved(session_id, all_chunks, synthesis_sub_queries)
        await websocket.send_json({
            'type': 'retrieved',
            'chunk_count': len(all_chunks),
            'latency_ms': 0.0,
            'source': 'filter'
        })

    else:
        # REQUERY: starvation guard — run a fresh retrieval with the original semantic query,
        # then apply metadata constraints post-retrieval.
        from core.state import apply_constraints as _apply_constraints
        retrieval_sub_queries = [SubQuery(id='sq_0', text=delta.augmented_query, intent_type='factual')]
        try:
            import time as _time
            t0 = _time.time()
            chunks_by_query = await retriever.retrieve_multi(retrieval_sub_queries)
            retr_ms = (_time.time() - t0) * 1000
            all_chunks = [c for cs in chunks_by_query.values() for c in cs]

            # Apply metadata constraints post-retrieval (preserves semantic quality)
            if session.constraints:
                all_chunks = _apply_constraints(all_chunks, session.constraints)
                chunks_by_query = {'sq_0': all_chunks}

            state_manager.mark_retrieved(session_id, all_chunks, sub_queries=None)
            await websocket.send_json({
                'type': 'retrieved',
                'chunk_count': len(all_chunks),
                'latency_ms': round(retr_ms, 1),
                'source': 'requery',
            })

            
            if len(all_chunks) == 0:
                msg = "I don't have enough evidence in the supplied corpus to answer this."
                state_manager.update_answer(session_id, msg, [])
                try:
                    await websocket.send_json({'type': 'token', 'content': msg})
                    await websocket.send_json({'type': 'insufficient_evidence', 'clarifying_question': ''})
                    await websocket.send_json({'type': 'done', 'latency_ms': 0.0, 'citation_count': 0})
                except Exception: pass
                logger.log_insufficient_evidence(session_id, session.turn_count)
                return
                
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.log_error(session_id, session.turn_count, f'REQUERY failed: {e}', True, 'RequeryError')
            await websocket.send_json({'type': 'error', 'message': f'REQUERY failed: {e}'})
            return

    await _handle_synthesis(
        websocket, session_id, session.full_transcript(), synthesis_sub_queries, chunks_by_query,
        refinement_suffix=delta.refinement_prompt_suffix,
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api.server:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
