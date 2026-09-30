"""
core/retriever.py
Hybrid retrieval pipeline: BM25 sparse + ChromaDB dense + RRF fusion + cross-encoder reranking.

Uses `transformers` + `torch` directly for embedding and reranking (avoids the
sklearn→scipy import chain which triggers WDAC DLL restrictions on some Windows setups).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from core import RetrievedChunk, SubQuery

load_dotenv()


# ── Lightweight embedding model (replaces SentenceTransformer) ───────────────

class _EmbedModel:
    """Mean-pooled dense encoder using HuggingFace transformers + torch."""

    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**encoded)
                # Mean pooling over token embeddings
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                emb = (outputs.last_hidden_state * attention_mask).sum(1)
                emb = emb / attention_mask.sum(1).clamp(min=1e-9)
                if normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


# ── Lightweight cross-encoder (replaces CrossEncoder) ───────────────────────

class _CrossEncoder:
    """Sequence-classification cross-encoder using HuggingFace transformers."""

    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> np.ndarray:
        all_scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries = [p[0] for p in batch]
            docs    = [p[1] for p in batch]
            encoded = self._tokenizer(
                queries, docs, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits
                # For binary relevance models: use logit of class 1, or squeeze
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1)
                else:
                    scores = logits[:, 1]          # positive-class logit
            all_scores.extend(scores.cpu().tolist())
        return np.array(all_scores, dtype=np.float32)


import threading

# ── HybridRetriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retrieval pipeline.

    1. Dense: ChromaDB + _EmbedModel (mean-pooled transformers)
    2. Sparse: BM25Okapi (rank_bm25)
    3. Fusion: Reciprocal Rank Fusion (RRF, k=60)
    4. Reranking: _CrossEncoder (transformers sequence classification)
    """

    def __init__(self):
        self._encode_lock = threading.Lock()
        self._rerank_lock = threading.Lock()
        
        embed_model   = os.environ.get("EMBED_MODEL",   "all-MiniLM-L6-v2")
        rerank_model  = os.environ.get("RERANK_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")
        chroma_mode   = os.environ.get("CHROMA_MODE",   "persist")
        chroma_path   = os.environ.get("CHROMA_PATH",   "./data/chroma_db")
        chroma_col    = os.environ.get("CHROMA_COLLECTION", "rag_corpus")

        self._retrieval_top_k       = int(os.environ.get("RETRIEVAL_TOP_K", 20))
        self._rerank_top_k          = int(os.environ.get("RERANK_TOP_K", 5))
        self._rrf_k                 = int(os.environ.get("RRF_K", 60))
        self._rerank_score_threshold = float(os.environ.get("RERANK_SCORE_THRESHOLD", -2.0))

        # HuggingFace model name for all-MiniLM-L6-v2
        hf_embed_name   = f"sentence-transformers/{embed_model}"
        hf_rerank_name  = rerank_model  # cross-encoder/ms-marco-MiniLM-L-6-v2

        self._embed_model = _EmbedModel(hf_embed_name)
        self._reranker    = _CrossEncoder(hf_rerank_name)

        if chroma_mode == "memory":
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.PersistentClient(path=chroma_path)

        self._collection = client.get_or_create_collection(
            name=chroma_col,
            metadata={"hnsw:space": "cosine"}
        )

        self._corpus_docs: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._doc_id_to_index: dict[str, int] = {}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _make_entry_id(self, doc: dict) -> str:
        return f"{doc['doc_id']}_{doc['section'].replace(' ', '_')}"

    def _build_bm25_index(self) -> None:
        tokenized = [doc["text"].lower().split() for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._doc_id_to_index = {
            self._make_entry_id(doc): i for i, doc in enumerate(self._corpus_docs)
        }

    def _rrf_fusion(
        self,
        dense_ids: list[str],  dense_scores: list[float],
        sparse_ids: list[str], sparse_scores: list[float],
    ) -> list[str]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d)) across both lists.
        Each document is scored exactly once per ranked list.
        """
        all_ids = set(dense_ids) | set(sparse_ids)
        dense_rank  = {doc_id: rank for rank, doc_id in enumerate(dense_ids,  start=1)}
        sparse_rank = {doc_id: rank for rank, doc_id in enumerate(sparse_ids, start=1)}
        absent_d = len(dense_ids)  + 1
        absent_s = len(sparse_ids) + 1

        scores: dict[str, float] = {}
        for doc_id in all_ids:
            dr = dense_rank.get(doc_id,  absent_d)
            sr = sparse_rank.get(doc_id, absent_s)
            scores[doc_id] = (1.0 / (self._rrf_k + dr)) + (1.0 / (self._rrf_k + sr))

        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # ── Public API ──────────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        loop = asyncio.get_running_loop()

        # 1. Dense embedding (run in executor to avoid blocking event loop)
        def _do_embed():
            with getattr(self, '_encode_lock', __import__('contextlib').nullcontext()):
                return self._embed_model.encode([query], normalize_embeddings=True)[0].tolist()

        embed = await loop.run_in_executor(None, _do_embed)

        # 2. Dense retrieval from ChromaDB
        dense_ids: list[str] = []
        dense_scores: list[float] = []
        count = self._collection.count()
        if count > 0:
            n_res = min(self._retrieval_top_k, count)
            result = self._collection.query(
                query_embeddings=[embed],
                n_results=n_res,
                include=["documents", "metadatas", "distances"],  # ids always returned by default
            )
            if result and result["ids"] and result["ids"][0]:
                dense_ids    = result["ids"][0]
                dense_scores = [1.0 - d for d in result["distances"][0]]  # cosine → similarity

        # 3. Sparse BM25 retrieval
        sparse_ids: list[str] = []
        sparse_scores: list[float] = []
        if self._bm25 and self._corpus_docs:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            top_indices = np.argsort(bm25_scores)[::-1][: self._retrieval_top_k]
            sparse_ids    = [self._make_entry_id(self._corpus_docs[i]) for i in top_indices]
            sparse_scores = [float(bm25_scores[i]) for i in top_indices]

        # 4. RRF fusion
        rrf_ids = self._rrf_fusion(dense_ids, dense_scores, sparse_ids, sparse_scores)

        # 5. Candidate lookup
        candidates_dict = {self._make_entry_id(d): d for d in self._corpus_docs}
        n_candidates = min(self._rerank_top_k * 3, len(candidates_dict))
        candidates = [
            candidates_dict[did] for did in rrf_ids[:n_candidates] if did in candidates_dict
        ]
        if not candidates:
            return []

        # 6. Cross-encoder reranking (run in executor)
        pairs = [[query, doc["text"]] for doc in candidates]
        
        def _do_rerank():
            with getattr(self, '_rerank_lock', __import__('contextlib').nullcontext()):
                return self._reranker.predict(pairs)
                
        cross_scores = await loop.run_in_executor(None, _do_rerank)

        scored = sorted(
            [(float(cross_scores[i]), candidates[i]) for i in range(len(candidates))],
            key=lambda x: x[0],
            reverse=True,
        )
        scored = [(s, d) for s, d in scored if s >= self._rerank_score_threshold]

        final_k = top_k if top_k is not None else self._rerank_top_k
        return [
            RetrievedChunk(
                doc_id=doc["doc_id"],
                section=doc["section"],
                text=doc["text"],
                score=score,
                rank=rank,
                metadata=doc.get("metadata", {}),
            )
            for rank, (score, doc) in enumerate(scored[:final_k], start=1)
        ]

    async def retrieve_multi(self, sub_queries: list[SubQuery]) -> dict[str, list[RetrievedChunk]]:
        results = await asyncio.gather(*[self.retrieve(sq.text) for sq in sub_queries])
        return {sq.id: chunks for sq, chunks in zip(sub_queries, results)}

    def index_corpus(self, docs: list[dict]) -> None:
        if not docs:
            return
        self._corpus_docs = docs
        self._build_bm25_index()

        ids       = [self._make_entry_id(d) for d in docs]
        texts     = [d["text"] for d in docs]
        metadatas = [
            {"doc_id": d["doc_id"], "section": d["section"], **d.get("metadata", {})}
            for d in docs
        ]
        embeddings = self._embed_model.encode(texts, normalize_embeddings=True).tolist()
        self._collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def get_corpus_stats(self) -> dict:
        return {
            "total_chunks":    self._collection.count(),
            "collection_name": self._collection.name,
            "bm25_vocab_size": len(self._bm25.idf) if self._bm25 else 0,
            "corpus_docs":     len(self._corpus_docs),
        }
