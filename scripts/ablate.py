import asyncio
import time
import json
import os
import sys
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.retriever import HybridRetriever
from core.decomposer import IntentDecomposer
from core.synthesizer import AnswerSynthesizer
from core import SubQuery, RetrievedChunk

# 20 Queries
EVAL_QUERIES = [
    # Factual (Direct)
    ("What are the key components of RAG architecture?", "ai_004"),
    ("Who introduced the Transformer architecture and when?", "ai_001"),
    ("What did the IPCC AR6 report say about 1.5C warming?", "cl_003"),
    ("When did the James Webb Space Telescope launch?", "sp_002"),
    ("How do mRNA vaccines work compared to traditional ones?", "med_001"),
    
    # Keyword-heavy
    ("HNSW algorithm approximate nearest neighbor search vector databases", "ai_008"),
    ("Direct Air Capture CO2 gigaton scale cost per ton", "cl_006"),
    ("NASA Perseverance rover Ingenuity helicopter organic molecule", "sp_001"),
    ("Event Horizon Telescope M87 Sagittarius A* accretion disk", "sp_006"),
    ("CRISPR Cas9 guide RNA targeted DNA cleavage", "med_002"),
    
    # Semantic (abstract, conversational)
    ("Can you explain how language models align with human preferences?", "ai_007"),
    ("I want to know why models sometimes make up fake information and how to stop it", "ai_010"),
    ("What happens if we push the earth's climate past points of no return?", "cl_007"),
    ("Are there any planets outside our solar system that might have life?", "sp_005"),
    ("Tell me about the protein folding AI by DeepMind", "med_003"),
    
    # Multi-intent / Compound
    ("What is fine-tuning and how does it relate to prompt engineering?", ["ai_006", "ai_009"]),
    ("Compare the greenhouse effect with ocean acidification", ["cl_001", "cl_009"]),
    ("What is the Artemis program and what rocket is it using?", "sp_003"),
    ("Explain dark matter and also tell me about solar flares", ["sp_007", "sp_009"]),
    ("How does immunotherapy fight cancer and what is antibiotic resistance?", ["med_005", "med_004"])
]

async def run_ablation_1():
    print("="*50)
    print("ABLATION 1 — HYBRID RETRIEVAL VS DENSE-ONLY")
    print("="*50)
    
    retriever = HybridRetriever()
    corpus_path = Path(__file__).parent.parent / 'data' / 'corpus.json'
    with open(corpus_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    retriever.index_corpus(docs)
    
    decomposer = IntentDecomposer()
    
    async def evaluate_retrieval(use_bm25=True):
        if not use_bm25:
            # Monkeypatch BM25 to None to disable sparse retrieval
            original_bm25 = retriever._bm25
            retriever._bm25 = None
            
        results = []
        for q, expected in EVAL_QUERIES:
            t0 = time.time()
            sub_queries = decomposer.deterministic_split(q)
            if not sub_queries:
                sub_queries = [SubQuery(id='sq_0', text=q, intent_type='factual')]
                
            chunks_by_query = await retriever.retrieve_multi(sub_queries)
            retr_ms = (time.time() - t0) * 1000
            
            all_chunks = [c for cs in chunks_by_query.values() for c in cs]
            all_chunks.sort(key=lambda x: x.rank)
            retrieved_doc_ids = [c.doc_id for c in all_chunks]
            
            expected_list = expected if isinstance(expected, list) else [expected]
            top_5 = retrieved_doc_ids[:5]
            top_10 = retrieved_doc_ids[:10]
            
            recall_5 = sum(1 for e in expected_list if e in top_5) / len(expected_list)
            recall_10 = sum(1 for e in expected_list if e in top_10) / len(expected_list)
            
            results.append({
                "query": q,
                "latency_ms": retr_ms,
                "recall_5": recall_5,
                "recall_10": recall_10
            })
            
        if not use_bm25:
            # Restore
            retriever._bm25 = original_bm25
            
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        avg_recall_5 = sum(r["recall_5"] for r in results) / len(results)
        avg_recall_10 = sum(r["recall_10"] for r in results) / len(results)
        
        return avg_recall_5, avg_recall_10, avg_latency

    # Warmup
    await retriever.retrieve("warmup")
    
    h_r5, h_r10, h_lat = await evaluate_retrieval(use_bm25=True)
    d_r5, d_r10, d_lat = await evaluate_retrieval(use_bm25=False)
    
    print("| Metric | Hybrid | Dense-only | Difference |")
    print("|---|---:|---:|---:|")
    print(f"| Recall@5 | {h_r5:.2f} | {d_r5:.2f} | {h_r5 - d_r5:+.2f} |")
    print(f"| Recall@10 | {h_r10:.2f} | {d_r10:.2f} | {h_r10 - d_r10:+.2f} |")
    print(f"| Avg retrieval latency | {h_lat:.1f}ms | {d_lat:.1f}ms | {h_lat - d_lat:+.1f}ms |")


async def run_ablation_2():
    print("\n" + "="*50)
    print("ABLATION 2 — RERANKING ON VS OFF")
    print("="*50)
    
    retriever = HybridRetriever()
    corpus_path = Path(__file__).parent.parent / 'data' / 'corpus.json'
    with open(corpus_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    retriever.index_corpus(docs)
    decomposer = IntentDecomposer()
    synthesizer = AnswerSynthesizer()
    
    class MockReranker:
        def predict(self, pairs):
            # Return scores that maintain the exact original order of candidates (which is RRF order)
            return np.arange(len(pairs), 0, -1, dtype=np.float32)

    async def evaluate_reranking(use_reranker=True, do_synthesis=False):
        original_reranker = retriever._reranker
        if not use_reranker:
            retriever._reranker = MockReranker()
            
        results = []
        synth_results = []
        
        for idx, (q, expected) in enumerate(EVAL_QUERIES):
            t0 = time.time()
            sub_queries = decomposer.deterministic_split(q)
            if not sub_queries:
                sub_queries = [SubQuery(id='sq_0', text=q, intent_type='factual')]
                
            chunks_by_query = await retriever.retrieve_multi(sub_queries)
            retr_ms = (time.time() - t0) * 1000
            
            all_chunks = [c for cs in chunks_by_query.values() for c in cs]
            all_chunks.sort(key=lambda x: x.rank)
            retrieved_doc_ids = [c.doc_id for c in all_chunks]
            
            expected_list = expected if isinstance(expected, list) else [expected]
            top_5 = retrieved_doc_ids[:5]
            top_10 = retrieved_doc_ids[:10]
            
            recall_5 = sum(1 for e in expected_list if e in top_5) / len(expected_list)
            recall_10 = sum(1 for e in expected_list if e in top_10) / len(expected_list)
            
            results.append({
                "latency_ms": retr_ms,
                "recall_5": recall_5,
                "recall_10": recall_10
            })
            
            if do_synthesis and idx < 1:
                async def on_token(t): pass
                async def on_citation(d, s): pass
                
                synth_data = {}
                async def on_done(res, cost=0.0):
                    synth_data['res'] = res
                
                s_t0 = time.time()
                try:
                    await synthesizer.synthesize_merged(sub_queries, chunks_by_query, on_token, on_citation, on_done)
                except Exception as e:
                    print(f"Exception: {e}")
                    
                s_lat = (time.time() - s_t0) * 1000
                
                # Check groundedness
                cits = synth_data.get('res').citations if getattr(synth_data.get('res'), 'citations', None) else []
                cit_text = " ".join(cits)
                grounded = sum(1 for e in expected_list if e in cit_text) / len(expected_list)
                
                synth_results.append({
                    "ttft_ms": synth_data.get('res').latency_ms if getattr(synth_data.get('res'), 'latency_ms', None) else 0.0,
                    "latency": s_lat,
                    "grounded": grounded
                })
            
        if not use_reranker:
            retriever._reranker = original_reranker
            
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        avg_recall_5 = sum(r["recall_5"] for r in results) / len(results)
        avg_recall_10 = sum(r["recall_10"] for r in results) / len(results)
        
        avg_ttft = 0
        avg_grounded = 0
        if do_synthesis:
            avg_ttft = sum(r["ttft_ms"] for r in synth_results) / len(synth_results)
            avg_grounded = sum(r["grounded"] for r in synth_results) / len(synth_results)
            
        return avg_recall_5, avg_recall_10, avg_latency, avg_ttft, avg_grounded

    # Warmup
    await retriever.retrieve("warmup")
    
    r_r5, r_r10, r_lat, r_ttft, r_grnd = await evaluate_reranking(use_reranker=True, do_synthesis=True)
    n_r5, n_r10, n_lat, n_ttft, n_grnd = await evaluate_reranking(use_reranker=False, do_synthesis=True)
    
    print("| Metric | Reranking ON | Reranking OFF | Difference |")
    print("|---|---:|---:|---:|")
    print(f"| Recall@5 | {r_r5:.2f} | {n_r5:.2f} | {r_r5 - n_r5:+.2f} |")
    print(f"| Recall@10 | {r_r10:.2f} | {n_r10:.2f} | {r_r10 - n_r10:+.2f} |")
    print(f"| Avg retrieval latency | {r_lat:.1f}ms | {n_lat:.1f}ms | {r_lat - n_lat:+.1f}ms |")
    print(f"| TTFT (subset) | {r_ttft:.1f}ms | {n_ttft:.1f}ms | {r_ttft - n_ttft:+.1f}ms |")
    print(f"| Groundedness (subset) | {r_grnd:.2f} | {n_grnd:.2f} | {r_grnd - n_grnd:+.2f} |")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    asyncio.run(run_ablation_1())
    asyncio.run(run_ablation_2())
