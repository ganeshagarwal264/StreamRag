import asyncio, time, json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from core.retriever import HybridRetriever
from core.synthesizer import AnswerSynthesizer
from core import SubQuery

async def main():
    print("==================================================")
    print("EXPERIMENT A - CONTEXT SIZE")
    print("==================================================")
    
    retriever = HybridRetriever()
    corpus_path = Path('data/corpus.json')
    with open(corpus_path) as f:
        docs = json.load(f)
    retriever.index_corpus(docs)
    
    synthesizer = AnswerSynthesizer()
    query = "What is retrieval augmented generation?"
    sub_query = SubQuery(id="1", text=query, intent_type="factual")
    
    from core import RetrievedChunk
    mock_chunks = [
        RetrievedChunk(doc_id="ai_003", section="RAG Overview", text=docs[2]['text'], score=0.9, rank=1, metadata={}),
        RetrievedChunk(doc_id="ai_004", section="RAG Architecture", text=docs[3]['text'], score=0.8, rank=2, metadata={}),
        RetrievedChunk(doc_id="ai_001", section="Transformer Architecture", text=docs[0]['text'], score=0.7, rank=3, metadata={}),
        RetrievedChunk(doc_id="ai_010", section="Hallucination", text=docs[9]['text'], score=0.6, rank=4, metadata={}),
        RetrievedChunk(doc_id="ai_008", section="Vector Databases", text=docs[7]['text'], score=0.5, rank=5, metadata={})
    ]
    
    # Warmup
    print("Warming up Ollama...")
    import litellm
    try:
        resp = await litellm.acompletion(
            model="ollama/gemma4:12b",
            messages=[{"role": "user", "content": "hello"}],
            stream=True, max_tokens=10
        )
        async for chunk in resp: pass
    except Exception: pass
    print("Warmup complete.")

    results = {}
    for size in [1, 3, 5]:
        print(f"\n--- Testing with {size} chunk(s) ---")
        chunks = mock_chunks[:size]
        
        for run in range(3):
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
                    [sub_query],
                    {"1": chunks},
                    on_token, on_citation, on_done
                )
                t_total = time.time() - t0
                gen_time = t_total - ttft if ttft else 0
                tok_sec = tokens / gen_time if gen_time > 0 else 0
                print(f"  Run {run+1}: TTFT {ttft*1000 if ttft else 0:.0f}ms | Total {t_total*1000:.0f}ms | {tokens} tokens | {tok_sec:.1f} tok/s")
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
