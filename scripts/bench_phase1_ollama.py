"""Phase 1: Direct Ollama baseline. No embedding/reranker models loaded."""
import asyncio, time, json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

async def main():
    import litellm
    
    print("==================================================")
    print("PHASE 1: DIRECT OLLAMA BASELINE (no embed models)")
    print("==================================================")
    
    # Warm-up run
    print("\n--- Warm-up ---")
    try:
        resp = await litellm.acompletion(
            model="ollama/gemma4:12b",
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True, temperature=0.1, max_tokens=20
        )
        async for chunk in resp:
            pass
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return
    
    # Direct Ollama: short prompt
    print("\n--- Short Prompt (no RAG context) ---")
    short_results = []
    for i in range(3):
        t0 = time.time()
        ttft = None
        tokens = 0
        full_text = ""
        try:
            resp = await litellm.acompletion(
                model="ollama/gemma4:12b",
                messages=[{"role": "user", "content": "What is retrieval augmented generation? Be concise."}],
                stream=True, temperature=0.2, max_tokens=256
            )
            async for chunk in resp:
                c = chunk.choices[0].delta.content
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
                    full_text += c
            t_total = time.time() - t0
            gen_time = t_total - ttft if ttft else 0
            tok_sec = tokens / gen_time if gen_time > 0 else 0
            r = {"run": i+1, "ttft_ms": round(ttft*1000, 1) if ttft else 0, 
                 "total_ms": round(t_total*1000, 1), "tokens": tokens, 
                 "gen_ms": round(gen_time*1000, 1), "tok_sec": round(tok_sec, 1)}
            print(f"  Run {i+1}: TTFT={r['ttft_ms']}ms  Gen={r['gen_ms']}ms  Total={r['total_ms']}ms  Tokens={tokens}  Tok/s={r['tok_sec']}")
            short_results.append(r)
        except Exception as e:
            print(f"  Run {i+1} FAILED: {e}")
    
    # Direct Ollama: with RAG-sized context
    print("\n--- Long Prompt (simulated RAG context, ~2000 tokens) ---")
    context = """
=== CONTEXT CHUNK [1] ===
Source: [ai_003 S RAG Overview]
Text: Retrieval-Augmented Generation (RAG) combines parametric LLM knowledge with non-parametric retrieval from external knowledge bases. This approach significantly reduces hallucinations by grounding the model's responses in retrieved evidence. RAG enables knowledge updates without expensive retraining by simply updating the retrieval corpus. The architecture consists of a retriever component that fetches relevant passages from a vector database, and a generator component that conditions its output on both the query and the retrieved context.
===========================

=== CONTEXT CHUNK [2] ===
Source: [ai_004 S RAG Architecture]  
Text: The RAG architecture has two main components: the retriever and the generator. The retriever uses dense embeddings to find k-nearest neighbors in a vector database such as ChromaDB, Qdrant, or Pinecone. The generator, typically a large language model, takes both the original query and the retrieved passages as input and produces a grounded answer. Cross-encoder reranking can be applied between retrieval and generation to improve precision. The system supports both single-hop and multi-hop reasoning patterns.
===========================

=== CONTEXT CHUNK [3] ===
Source: [ai_001 S Transformer Architecture]
Text: Transformers were introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. The architecture replaces recurrent processing with self-attention mechanisms that process all input tokens simultaneously as key-query-value triplets. Multi-head attention allows the model to attend to different positions in the sequence from different representational subspaces. This parallel processing capability was a major breakthrough over the sequential nature of RNNs and LSTMs.
===========================

=== CONTEXT CHUNK [4] ===
Source: [ai_010 S Hallucination]
Text: LLM hallucination occurs when models generate plausible but factually incorrect information. RAG reduces hallucination by grounding responses in retrieved evidence from trusted sources. Citation requirements enforce accountability by requiring every factual claim to reference its source document. Retrieval-augmented approaches have been shown to significantly improve factual accuracy compared to pure parametric generation.
===========================

=== CONTEXT CHUNK [5] ===
Source: [ai_008 S Vector Databases]
Text: Vector databases store high-dimensional embeddings and enable efficient similarity search at scale. The HNSW (Hierarchical Navigable Small World) algorithm provides approximate nearest-neighbor search with sub-linear time complexity. Popular vector database solutions include ChromaDB, Qdrant, Pinecone, and Weaviate. These databases form the backbone of RAG systems, enabling fast retrieval of relevant passages for language model conditioning.
===========================
"""
    system_prompt = """You are a precise, evidence-bound answering system. You MUST follow these rules STRICTLY:
1. Answer ONLY using the provided context chunks.
2. Every factual assertion MUST be followed immediately by its citation in the format [doc_id S section].
3. If the provided context does not contain sufficient evidence, output exactly: INSUFFICIENT_EVIDENCE
4. Be concise and precise."""

    long_results = []
    for i in range(3):
        t0 = time.time()
        ttft = None
        tokens = 0
        try:
            resp = await litellm.acompletion(
                model="ollama/gemma4:12b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: What is retrieval augmented generation?"}
                ],
                stream=True, temperature=0.2, max_tokens=1024
            )
            async for chunk in resp:
                c = chunk.choices[0].delta.content
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
            t_total = time.time() - t0
            gen_time = t_total - ttft if ttft else 0
            tok_sec = tokens / gen_time if gen_time > 0 else 0
            r = {"run": i+1, "ttft_ms": round(ttft*1000, 1) if ttft else 0, 
                 "total_ms": round(t_total*1000, 1), "tokens": tokens, 
                 "gen_ms": round(gen_time*1000, 1), "tok_sec": round(tok_sec, 1)}
            print(f"  Run {i+1}: TTFT={r['ttft_ms']}ms  Gen={r['gen_ms']}ms  Total={r['total_ms']}ms  Tokens={tokens}  Tok/s={r['tok_sec']}")
            long_results.append(r)
        except Exception as e:
            print(f"  Run {i+1} FAILED: {e}")
    
    # Summary
    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    if short_results:
        ttfts = [r['ttft_ms'] for r in short_results]
        totals = [r['total_ms'] for r in short_results]
        toks = [r['tok_sec'] for r in short_results]
        print(f"Short prompt TTFT:  min={min(ttfts):.0f}ms  max={max(ttfts):.0f}ms  mean={sum(ttfts)/len(ttfts):.0f}ms")
        print(f"Short prompt total: min={min(totals):.0f}ms  max={max(totals):.0f}ms  mean={sum(totals)/len(totals):.0f}ms")
        print(f"Short prompt tok/s: min={min(toks):.1f}  max={max(toks):.1f}  mean={sum(toks)/len(toks):.1f}")
    if long_results:
        ttfts = [r['ttft_ms'] for r in long_results]
        totals = [r['total_ms'] for r in long_results]
        toks = [r['tok_sec'] for r in long_results]
        print(f"Long prompt TTFT:   min={min(ttfts):.0f}ms  max={max(ttfts):.0f}ms  mean={sum(ttfts)/len(ttfts):.0f}ms")
        print(f"Long prompt total:  min={min(totals):.0f}ms  max={max(totals):.0f}ms  mean={sum(totals)/len(totals):.0f}ms")
        print(f"Long prompt tok/s:  min={min(toks):.1f}  max={max(toks):.1f}  mean={sum(toks)/len(toks):.1f}")

asyncio.run(main())
