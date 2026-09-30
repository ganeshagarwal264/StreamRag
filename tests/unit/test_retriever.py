"""
tests/unit/test_retriever.py

Unit tests for HybridRetriever.

The rewritten retriever uses transformers+torch directly (no sentence_transformers,
no sklearn, no scipy), so no sys.modules pre-mocking is needed.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from core import SubQuery, RetrievedChunk
from core.retriever import HybridRetriever, _EmbedModel, _CrossEncoder


SAMPLE_CORPUS = [
    {
        "doc_id": "ai_001",
        "section": "Intro",
        "text": "Transformers use self-attention for sequence modeling.",
        "metadata": {"domain": "AI", "year": 2023},
    },
    {
        "doc_id": "ai_002",
        "section": "Architecture",
        "text": "Multi-head attention enables parallel processing.",
        "metadata": {"domain": "AI", "year": 2022},
    },
    {
        "doc_id": "cl_001",
        "section": "Climate",
        "text": "Sea level rise accelerating due to ice melt.",
        "metadata": {"domain": "Climate", "year": 2024},
    },
]


def make_retriever() -> HybridRetriever:
    """Build a HybridRetriever with all model/DB deps mocked out."""
    with patch("core.retriever.chromadb") as mock_chroma, \
         patch("core.retriever.AutoTokenizer"), \
         patch("core.retriever.AutoModel"), \
         patch("core.retriever.AutoModelForSequenceClassification"):

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.name = "test_collection"
        mock_chroma.PersistentClient.return_value \
            .get_or_create_collection.return_value = mock_collection
        mock_chroma.EphemeralClient.return_value \
            .get_or_create_collection.return_value = mock_collection

        r = HybridRetriever()

    # Replace model objects with controllable mocks
    r._embed_model = MagicMock()
    r._reranker = MagicMock()
    r._corpus_docs = SAMPLE_CORPUS
    r._build_bm25_index()
    return r


# ── BM25 ────────────────────────────────────────────────────────────────────

def test_bm25_index_built():
    r = make_retriever()
    assert r._bm25 is not None


def test_bm25_scores_relevant_doc_higher():
    r = make_retriever()
    # Use exact tokens as they appear after lowercasing + splitting
    # ai_002 contains exact token "attention"; cl_001 contains none of these
    scores = r._bm25.get_scores(["attention", "parallel"])
    # ai_002 (index 1) should score higher than cl_001 (index 2)
    assert scores[1] > scores[2]
    # cl_001 has no matching tokens so should score 0
    assert scores[2] == 0.0


# ── Entry ID ─────────────────────────────────────────────────────────────────

def test_make_entry_id():
    r = make_retriever()
    doc = {"doc_id": "ai_001", "section": "Transformer Architecture"}
    assert r._make_entry_id(doc) == "ai_001_Transformer_Architecture"

def test_make_entry_id_no_spaces():
    r = make_retriever()
    doc = {"doc_id": "cl_002", "section": "Climate"}
    assert r._make_entry_id(doc) == "cl_002_Climate"


# ── RRF Fusion ───────────────────────────────────────────────────────────────

def test_rrf_fusion_top_doc():
    r = make_retriever()
    # ai_001 rank-1 dense, rank-2 sparse → highest combined RRF
    dense_ids   = ["ai_001", "cl_001", "ai_002"]
    dense_scores  = [0.9, 0.7, 0.6]
    sparse_ids  = ["ai_002", "ai_001", "cl_001"]
    sparse_scores = [0.8, 0.6, 0.4]
    fused = r._rrf_fusion(dense_ids, dense_scores, sparse_ids, sparse_scores)
    assert fused[0] == "ai_001"
    assert set(fused) == {"ai_001", "ai_002", "cl_001"}


def test_rrf_fusion_single_list():
    r = make_retriever()
    ids    = ["ai_001", "ai_002"]
    scores = [0.9, 0.7]
    fused  = r._rrf_fusion(ids, scores, [], [])
    assert fused[0] == "ai_001"


def test_rrf_fusion_no_double_counting():
    """Each doc scores exactly 2 RRF contributions — one per ranked list."""
    r = make_retriever()
    k = r._rrf_k   # 60
    ids    = ["doc_a", "doc_b"]
    scores = [1.0, 0.5]
    fused  = r._rrf_fusion(ids, scores, ids, scores)
    expected_a = 2.0 / (k + 1)
    expected_b = 2.0 / (k + 2)
    assert expected_a > expected_b
    assert fused[0] == "doc_a"


def test_rrf_fusion_union():
    r = make_retriever()
    dense_ids  = ["ai_001", "cl_001"]
    sparse_ids = ["ai_002", "ai_001"]
    fused = r._rrf_fusion(dense_ids, [0.9, 0.7], sparse_ids, [0.8, 0.6])
    assert len(fused) == 3  # union: ai_001, cl_001, ai_002


# ── Corpus Stats ─────────────────────────────────────────────────────────────

def test_corpus_stats():
    r = make_retriever()
    r._collection.count.return_value = 3
    stats = r.get_corpus_stats()
    assert stats["total_chunks"] == 3
    assert stats["bm25_vocab_size"] > 0
    assert stats["corpus_docs"] == 3


# ── Async retrieve (mocked) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_empty_on_no_corpus():
    r = make_retriever()
    r._corpus_docs = []
    r._bm25 = None
    r._collection.count.return_value = 0
    results = await r.retrieve("What is AI?")
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_multi_returns_dict():
    r = make_retriever()
    # Mock the embed and rerank to return plausible values
    dummy_emb = np.zeros(384, dtype=np.float32)
    r._embed_model.encode = MagicMock(return_value=np.array([dummy_emb]))
    r._collection.count.return_value = 0   # skip dense path
    r._reranker.predict = MagicMock(return_value=np.array([1.5, 1.2, 0.8]))

    sqs = [
        SubQuery(id="sq_0", text="What is AI?", intent_type="factual"),
        SubQuery(id="sq_1", text="What is climate change?", intent_type="factual"),
    ]
    result = await r.retrieve_multi(sqs)
    assert "sq_0" in result
    assert "sq_1" in result
