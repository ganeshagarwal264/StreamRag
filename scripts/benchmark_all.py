import asyncio
import time
import os
import sys
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.controller import IntentController
from core.decomposer import IntentDecomposer
from core.retriever import HybridRetriever
from core.synthesizer import AnswerSynthesizer
from core import SubQuery, SessionState

load_dotenv()

async def benchmark_direct_ollama():
    import litellm
    print("==================================================")
    print("3. DIRECT OLLAMA BASELINE")
    print("==================================================")
    results = []
    for i in range(3):
        t0 = time.time()
        try:
            resp = await litellm.acompletion(
                model="ollama/gemma4:12b",
                messages=[{"role": "user", "content": "What is retrieval augmented generation?"}],
                stream=True
            )
            ttft = None
            tokens = 0
            async for chunk in resp:
                if ttft is None and chunk.choices[0].delta.content:
                    ttft = time.time() - t0
                if chunk.choices[0].delta.content:
                    tokens += 1
            t_total = time.time() - t0
            tok_sec = tokens / (t_total - ttft) if ttft and (t_total - ttft) > 0 else 0
            res = {'ttft': ttft*1000 if ttft else 0, 'total': t_total*1000, 'tokens': tokens, 'tok_sec': tok_sec}
            print(f"Run {i+1}: {res}")
            results.append(res)
        except Exception as e:
            print(f"Run {i+1} Failed: {e}")
    return results

async def benchmark_components():
    print("\n==================================================")
    print("INITIALIZING COMPONENTS")
    print("==================================================")
    retriever = HybridRetriever()
    corpus_path = Path('data/corpus.json')
    if corpus_path.exists():
        with open(corpus_path) as f:
            docs = json.load(f)
        retriever.index_corpus(docs)
    controller = IntentController()
    decomposer = IntentDecomposer()
    synthesizer = AnswerSynthesizer()

    print("\n==================================================")
    print("8. DECOMPOSER")
    print("==================================================")
    t0 = time.time()
    sq1 = await decomposer.decompose("What is retrieval augmented generation?")
    d1 = time.time() - t0
    print(f"Simple (bypass expected): {d1*1000:.1f}ms, {len(sq1)} subqueries")
    
    t0 = time.time()
    sq2 = await decomposer.decompose("What is retrieval augmented generation and what are the main causes of climate change?")
    d2 = time.time() - t0
    print(f"Compound (LLM expected): {d2*1000:.1f}ms, {len(sq2)} subqueries")

    print("\n==================================================")
    print("7. RETRIEVAL COMPONENT BREAKDOWN")
    print("==================================================")
    query = "What is retrieval augmented generation?"
    t0 = time.time()
    embed = await asyncio.get_event_loop().run_in_executor(None, lambda: retriever._embed_model.encode([query], normalize_embeddings=True)[0].tolist())
    t_embed = time.time() - t0

    t0 = time.time()
    chroma_res = retriever._collection.query(query_embeddings=[embed], n_results=20)
    t_chroma = time.time() - t0

    t0 = time.time()
    sparse_scores = retriever._bm25.get_scores(query.lower().split())
    t_bm25 = time.time() - t0

    print(f"Embedding: {t_embed*1000:.1f}ms")
    print(f"Chroma: {t_chroma*1000:.1f}ms")
    print(f"BM25: {t_bm25*1000:.1f}ms")

    t0 = time.time()
    chunks = await retriever.retrieve_multi([SubQuery(id="1", text=query, intent_type="factual")])
    t_ret_total = time.time() - t0
    print(f"Total Retrieve_multi (incl cross-encoder): {t_ret_total*1000:.1f}ms")
    print(f"Chunks retrieved: {len(chunks['1'])}")

    print("\n==================================================")
    print("4 & 9. RAG SYNTHESIS BASELINE (5 runs)")
    print("==================================================")
    rag_results = []
    all_chunks = chunks["1"]
    for i in range(5):
        t0 = time.time()
        ttft = None
        tokens = 0
        async def on_token(t):
            nonlocal ttft, tokens
            if ttft is None:
                ttft = time.time() - t0
            tokens += 1
        async def on_citation(d, s): pass
        async def on_done(res, cost): pass
        
        try:
            await synthesizer.synthesize_merged(
                [SubQuery(id="1", text=query, intent_type="factual")],
                {"1": all_chunks},
                on_token, on_citation, on_done
            )
            t_total = time.time() - t0
            tok_sec = tokens / (t_total - ttft) if ttft and (t_total - ttft) > 0 else 0
            res = {'ttft': ttft*1000 if ttft else 0, 'total': t_total*1000, 'tokens': tokens, 'tok_sec': tok_sec}
            print(f"Run {i+1}: {res}")
            rag_results.append(res)
        except Exception as e:
            print(f"Run {i+1} Failed: {e}")

    print("\n==================================================")
    print("5. PROMPT SIZE EXPERIMENT")
    print("==================================================")
    for size in [1, 5]:
        if size > len(all_chunks):
            print(f"Skipping size {size}, only have {len(all_chunks)} chunks")
            continue
        test_chunks = all_chunks[:size]
        t0 = time.time()
        ttft = None
        async def on_token(t):
            nonlocal ttft
            if ttft is None:
                ttft = time.time() - t0
        async def on_citation(d, s): pass
        async def on_done(res, cost): pass
        
        try:
            await synthesizer.synthesize_merged(
                [SubQuery(id="1", text=query, intent_type="factual")],
                {"1": test_chunks},
                on_token, on_citation, on_done
            )
            t_total = time.time() - t0
            print(f"Prompt Size {size} chunks: TTFT {ttft*1000 if ttft else 0:.1f}ms, Total {t_total*1000:.1f}ms")
        except Exception as e:
            print(f"Prompt Size {size} Failed: {e}")

async def main():
    try:
        await benchmark_direct_ollama()
        await benchmark_components()
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
