"""Phase 2: Retrieval benchmark. No Ollama calls."""
import asyncio, time, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from core.retriever import HybridRetriever
from core.controller import IntentController
from core.decomposer import IntentDecomposer
from core import SubQuery, SessionState

async def main():
    print("==================================================")
    print("PHASE 2: RETRIEVAL + CONTROLLER + DECOMPOSER")
    print("(No Ollama calls)")
    print("==================================================")
    
    retriever = HybridRetriever()
    corpus_path = Path('data/corpus.json')
    with open(corpus_path) as f:
        docs = json.load(f)
    retriever.index_corpus(docs)
    print(f"Corpus loaded: {len(docs)} chunks")
    
    controller = IntentController()
    decomposer = IntentDecomposer()
    
    # --- Controller ---
    print("\n--- Controller Benchmark ---")
    queries = [
        ("What is retrieval augmented generation?", "RETRIEVE"),
        ("Who was Vaswani?", "RETRIEVE"),
        ("um uh well so", "WAIT"),
        ("next slide please", "SUPPRESS"),
    ]
    for q, expected in queries:
        session = SessionState(session_id="bench")
        t0 = time.time()
        d = await controller.assess(q, session)
        dt = (time.time() - t0) * 1000
        print(f"  '{q[:50]}...' -> {d.action} ({dt:.1f}ms) [expected {expected}]")
    
    # --- Decomposer ---
    print("\n--- Decomposer Benchmark ---")
    simple_q = "What is retrieval augmented generation?"
    t0 = time.time()
    sq1 = await decomposer.decompose(simple_q)
    dt1 = (time.time() - t0) * 1000
    print(f"  Simple: '{simple_q[:50]}' -> {len(sq1)} subqueries, {dt1:.1f}ms (bypass expected)")
    
    compound_q = "What is retrieval augmented generation and what are the main causes of climate change?"
    t0 = time.time()
    sq2 = await decomposer.decompose(compound_q)
    dt2 = (time.time() - t0) * 1000
    print(f"  Compound: '{compound_q[:50]}...' -> {len(sq2)} subqueries, {dt2:.1f}ms")
    
    # --- Retrieval Breakdown ---
    print("\n--- Retrieval Component Breakdown ---")
    query = "What is retrieval augmented generation?"
    loop = asyncio.get_event_loop()
    
    # Embedding
    t0 = time.time()
    embed = await loop.run_in_executor(None, lambda: retriever._embed_model.encode([query], normalize_embeddings=True)[0].tolist())
    t_embed = (time.time() - t0) * 1000
    
    # Dense (Chroma)
    count = retriever._collection.count()
    n = min(20, count)
    t0 = time.time()
    chroma_res = retriever._collection.query(query_embeddings=[embed], n_results=n)
    t_chroma = (time.time() - t0) * 1000
    
    # BM25
    t0 = time.time()
    bm25_scores = retriever._bm25.get_scores(query.lower().split())
    t_bm25 = (time.time() - t0) * 1000
    
    # Full retrieval (includes RRF + reranking)
    retr_results = []
    for i in range(5):
        t0 = time.time()
        chunks = await retriever.retrieve(query)
        dt = (time.time() - t0) * 1000
        retr_results.append({"run": i+1, "latency_ms": round(dt, 1), "chunks": len(chunks)})
        print(f"  Full retrieve run {i+1}: {dt:.1f}ms, {len(chunks)} chunks")
    
    # Reranking isolation
    import numpy as np
    dense_ids = chroma_res['ids'][0] if chroma_res['ids'] else []
    # Get candidate texts for reranking
    candidate_texts = []
    candidate_ids = []
    for did in dense_ids[:15]:
        for doc in docs:
            eid = f"{doc['doc_id']}_{doc['section'].replace(' ', '_')}"
            if eid == did:
                candidate_texts.append(doc['text'])
                candidate_ids.append(did)
                break
    
    if candidate_texts:
        pairs = [[query, t] for t in candidate_texts]
        t0 = time.time()
        scores = await loop.run_in_executor(None, lambda: retriever._reranker.predict(pairs))
        t_rerank = (time.time() - t0) * 1000
    else:
        t_rerank = 0
    
    print(f"\n  Component Breakdown:")
    print(f"    Embedding:     {t_embed:.1f}ms")
    print(f"    Chroma dense:  {t_chroma:.1f}ms")
    print(f"    BM25 sparse:   {t_bm25:.1f}ms")
    print(f"    Cross-encoder: {t_rerank:.1f}ms ({len(candidate_texts)} pairs)")
    
    retr_lats = [r['latency_ms'] for r in retr_results]
    print(f"\n  Full Retrieval (5 runs):")
    print(f"    Min:    {min(retr_lats):.1f}ms")
    print(f"    Max:    {max(retr_lats):.1f}ms")
    print(f"    Mean:   {sum(retr_lats)/len(retr_lats):.1f}ms")
    print(f"    Chunks: {retr_results[0]['chunks']}")

asyncio.run(main())
