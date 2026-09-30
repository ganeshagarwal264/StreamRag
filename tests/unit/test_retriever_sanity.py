"""
tests/unit/test_retriever_sanity.py

Pre-flight correctness tests for the hand-rolled embedding + cross-encoder
pipeline (transformers + torch, no sentence_transformers).

These tests exist to catch silent regressions in:
  - Mean-pooling strategy
  - L2 normalisation
  - Cross-encoder positive-class logit extraction

If either test fails, recall numbers from measure_metrics.py are UNRELIABLE.
Run this suite before trusting any retrieval metric.
"""
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

from core.retriever import HybridRetriever, _EmbedModel, _CrossEncoder


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_real_embed_model() -> _EmbedModel:
    """Load the real all-MiniLM-L6-v2 encoder (cached after first run)."""
    return _EmbedModel("sentence-transformers/all-MiniLM-L6-v2")


def make_real_cross_encoder() -> _CrossEncoder:
    """Load the real ms-marco cross-encoder (cached after first run)."""
    return _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cosine(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


# ── Fix 2 Sanity Tests ───────────────────────────────────────────────────────

def test_embedding_ordering_sanity():
    """
    A semantically similar pair must score higher than a dissimilar pair.

    If this fails the pooling strategy or normalisation is broken — DO NOT
    trust any recall numbers until this passes.
    """
    model = make_real_embed_model()
    embs = model.encode(
        [
            "What is a neural network?",      # query
            "Neural networks are ML models.",  # similar
            "Mars has two moons.",             # dissimilar
        ],
        normalize_embeddings=True,
    )
    query, similar, dissimilar = embs[0], embs[1], embs[2]

    sim_score   = cosine(query, similar)
    dissim_score = cosine(query, dissimilar)

    assert sim_score > dissim_score, (
        f"Embedding pipeline is not distinguishing relevant from irrelevant text — "
        f"sim={sim_score:.4f} <= dissim={dissim_score:.4f}. "
        f"Check pooling strategy and normalisation before trusting recall metrics."
    )


def test_embedding_normalisation():
    """Encoded vectors must have unit L2 norm."""
    import numpy as np
    model = make_real_embed_model()
    emb = model.encode(["hello world"], normalize_embeddings=True)
    norm = float(np.linalg.norm(emb[0]))
    assert abs(norm - 1.0) < 1e-4, (
        f"Embedding is not unit-normalised (norm={norm:.6f}). "
        "Cosine similarity will be wrong."
    )


def test_cross_encoder_sanity():
    """
    Cross-encoder must rank a directly relevant chunk above a same-domain
    but topically irrelevant chunk.

    If this fails the logit extraction or model loading is broken.
    """
    reranker = make_real_cross_encoder()
    query     = "When was the Mars rover launched?"
    relevant  = "Perseverance rover launched in July 2020 and landed February 2021."
    irrelevant = "Space debris in low Earth orbit poses collision risks to satellites."

    scores = reranker.predict([[query, relevant], [query, irrelevant]])
    rel_score  = float(scores[0])
    irrel_score = float(scores[1])

    assert rel_score > irrel_score, (
        f"Cross-encoder is not ranking relevant above irrelevant: "
        f"relevant={rel_score:.4f}, irrelevant={irrel_score:.4f}. "
        "Check logit extraction (binary vs scalar head)."
    )


def test_cross_encoder_returns_float_array():
    """predict() must return a numpy float array of length == len(pairs)."""
    import numpy as np
    reranker = make_real_cross_encoder()
    pairs = [["query one", "doc one"], ["query two", "doc two"]]
    scores = reranker.predict(pairs)
    assert isinstance(scores, np.ndarray), "predict() must return numpy.ndarray"
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert scores.dtype in (np.float32, np.float64)


def test_embedding_batch_consistency():
    """
    Encoding two sentences together must give the same vectors as encoding
    them individually (mean-pooling must be padding-mask-aware).
    """
    import numpy as np
    model = make_real_embed_model()

    s1, s2 = "Short sentence.", "A much longer sentence that has more tokens in it."
    batch = model.encode([s1, s2], normalize_embeddings=True)
    solo1 = model.encode([s1], normalize_embeddings=True)[0]
    solo2 = model.encode([s2], normalize_embeddings=True)[0]

    assert np.allclose(batch[0], solo1, atol=1e-4), (
        "Batch embedding[0] differs from solo encoding — "
        "pooling is not mask-aware; padding tokens are leaking."
    )
    assert np.allclose(batch[1], solo2, atol=1e-4), (
        "Batch embedding[1] differs from solo encoding — "
        "pooling is not mask-aware; padding tokens are leaking."
    )
