"""
tests/integration/test_concurrency.py
N concurrent sessions and retrieval calls must be safe.
"""
import asyncio, json, pytest
from pathlib import Path
from core.state import StateManager
from core.retriever import HybridRetriever

@pytest.mark.asyncio
async def test_session_state_no_leakage_under_concurrency():
    sm = StateManager()
    async def run(i):
        sid = f"conc_{i}"
        sm.append_token(sid, f"tok_{i}")
        await asyncio.sleep(0)
        assert sm.get_or_create(sid).tokens == [f"tok_{i}"], \
            f"Session {sid} leaked tokens: {sm.get_or_create(sid).tokens}"
        sm.destroy(sid)
    await asyncio.gather(*[run(i) for i in range(20)])

@pytest.mark.asyncio
async def test_concurrent_retrieve_no_crash():
    p = Path("data/corpus.json")
    if not p.exists(): pytest.skip("Run generate_mock_data.py first")
    r = HybridRetriever()
    r.index_corpus(json.loads(p.read_text(encoding="utf-8")))
    queries = ["transformer", "climate change", "Mars rover", "mRNA vaccine"]
    results = await asyncio.gather(*[r.retrieve(q) for q in queries])
    assert all(isinstance(res, list) for res in results)

@pytest.mark.asyncio
async def test_concurrent_sessions_return_independent_results():
    from core import SubQuery
    p = Path("data/corpus.json")
    if not p.exists(): pytest.skip("Run generate_mock_data.py first")
    r = HybridRetriever()
    r.index_corpus(json.loads(p.read_text(encoding="utf-8")))
    sqs_a = [SubQuery(id="sq_0", text="transformer attention", intent_type="factual")]
    sqs_b = [SubQuery(id="sq_0", text="renewable solar wind energy", intent_type="factual")]
    ra, rb = await asyncio.gather(r.retrieve_multi(sqs_a), r.retrieve_multi(sqs_b))
    ids_a = {c.doc_id for cs in ra.values() for c in cs}
    ids_b = {c.doc_id for cs in rb.values() for c in cs}
    assert ids_a != ids_b, "Different queries returned identical results — possible state leak"
