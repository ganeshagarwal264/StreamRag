import asyncio, time, json, sys, os, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from core.synthesizer import AnswerSynthesizer
from core import SubQuery, RetrievedChunk

async def run_baseline(docs):
    from core.retriever import HybridRetriever
    print("\n--- BASELINE (Retriever models resident in VRAM) ---")
    retriever = HybridRetriever()
    retriever.index_corpus(docs)
    
    synthesizer = AnswerSynthesizer()
    query = "What is retrieval augmented generation?"
    sub_query = SubQuery(id="1", text=query, intent_type="factual")
    
    chunks = [
        RetrievedChunk(doc_id="ai_003", section="RAG Overview", text=docs[2]['text'], score=0.9, rank=1, metadata={}),
        RetrievedChunk(doc_id="ai_004", section="RAG Architecture", text=docs[3]['text'], score=0.8, rank=2, metadata={}),
        RetrievedChunk(doc_id="ai_001", section="Transformer Architecture", text=docs[0]['text'], score=0.7, rank=3, metadata={})
    ]
    
    t0 = time.time()
    ttft = None
    async def on_token(t):
        nonlocal ttft
        if ttft is None: ttft = time.time() - t0
    async def on_citation(d, s): pass
    async def on_done(res, cost): pass
    
    await synthesizer.synthesize_merged([sub_query], {"1": chunks}, on_token, on_citation, on_done)
    print(f"  Run Baseline: TTFT {ttft*1000 if ttft else 0:.0f}ms")

async def run_experiment(docs):
    print("\n--- EXPERIMENT (Retriever models cleared from VRAM) ---")
    
    def fetch_chunks():
        from core.retriever import HybridRetriever
        retriever = HybridRetriever()
        retriever.index_corpus(docs)
        chunks = [
            RetrievedChunk(doc_id="ai_003", section="RAG Overview", text=docs[2]['text'], score=0.9, rank=1, metadata={}),
            RetrievedChunk(doc_id="ai_004", section="RAG Architecture", text=docs[3]['text'], score=0.8, rank=2, metadata={}),
            RetrievedChunk(doc_id="ai_001", section="Transformer Architecture", text=docs[0]['text'], score=0.7, rank=3, metadata={})
        ]
        return chunks
    
    chunks = fetch_chunks()
    
    print("  Triggering GC and clearing CUDA cache...")
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        print("  CUDA cache cleared.")
    except Exception as e:
        print(f"  Could not clear CUDA cache: {e}")
        
    synthesizer = AnswerSynthesizer()
    query = "What is retrieval augmented generation?"
    sub_query = SubQuery(id="1", text=query, intent_type="factual")
    
    t0 = time.time()
    ttft = None
    async def on_token(t):
        nonlocal ttft
        if ttft is None: ttft = time.time() - t0
    async def on_citation(d, s): pass
    async def on_done(res, cost): pass
    
    await synthesizer.synthesize_merged([sub_query], {"1": chunks}, on_token, on_citation, on_done)
    print(f"  Run Experiment: TTFT {ttft*1000 if ttft else 0:.0f}ms")

async def main():
    print("==================================================")
    print("EXPERIMENT B - GPU MEMORY RELEASE (1 rep each, 3 chunks)")
    print("==================================================")
    
    corpus_path = Path('data/corpus.json')
    with open(corpus_path) as f:
        docs = json.load(f)
        
    await run_baseline(docs)
    
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except: pass
        
    await run_experiment(docs)

if __name__ == "__main__":
    asyncio.run(main())
