# Streaming Live RAG - Architecture Brief

## 1. System Overview
The Streaming Live RAG system is a real-time, websocket-driven Retrieval-Augmented Generation application designed to process voice-transcription tokens as they arrive. The architecture guarantees factual grounding, provides multi-intent resolution, supports late-arriving conversational constraints, and performs speculative early retrieval to minimize time-to-first-token (TTFT).

## 2. Core Components

### 2.1 Intent Controller (`core/controller.py`)
Analyzes a stream of text tokens to determine conversational stability. It uses a hybrid rules-and-LLM approach:
- **Heuristics:** Checks for minimum token counts, dangling conjunctions (e.g., "and", "but"), and incomplete punctuation.
- **Navigational Commands:** Hardcoded suppression of UI navigation phrases (e.g., "next slide", "go back").
- **LLM Fallback:** If rules are indeterminate, it delegates to a fast LiteLLM call for a WAIT/RETRIEVE/SUPPRESS decision.

### 2.2 Intent Decomposer (`core/decomposer.py`)
Converts a compound user utterance into N atomic, self-contained sub-queries. It uses a structured LLM prompt (and a deterministic string-split fallback) to emit `SubQuery` objects (e.g., "What is RAG and how does it relate to transformers?" splits into two parallel vectors).

### 2.3 Hybrid Retriever (`core/retriever.py`)
Executes parallel retrieval across all sub-queries:
1. **Dense Retrieval:** Mean-pooled `all-MiniLM-L6-v2` against ChromaDB.
2. **Sparse Retrieval:** `BM25Okapi` sparse scoring.
3. **Fusion:** Reciprocal Rank Fusion (RRF) merges dense and sparse rankings.
4. **Reranking:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the top RRF candidates to yield the final `RetrievedChunk` array.

### 2.4 State & Refinement Engine (`core/state.py`)
Maintains an ephemeral, per-session state. It dynamically monitors incoming tokens for conversational constraints (e.g., "after 2020", "about AI", "most recent"). When a late constraint arrives during or after a generation, it computes a `DeltaInstruction`:
- **FILTER Mode:** If sufficient existing chunks satisfy the constraint, it immediately re-synthesizes using the filtered subset (bypassing the database).
- **REQUERY Mode:** If all chunks are excluded (starvation), it runs a fresh retrieval using the original semantic query, then applies the constraints post-retrieval.

### 2.5 Synthesizer (`core/synthesizer.py`)
The generator strictly grounds its output in the retrieved chunks. It enforces inline citations (e.g., `[ai_004 §RAG Architecture]`). If no evidence satisfies the query (or constraint), it emits a definitive `INSUFFICIENT_EVIDENCE` sentinel rather than hallucinating. It operates via asynchronous streaming (`litellm.acompletion`), handling network timeouts and retries gracefully.

## 3. Data Flow
1. **Token Ingestion:** Tokens stream via WebSocket to `api/server.py`.
2. **Early Retrieval (G2):** Debounced heuristic checks trigger speculative retrieval mid-utterance to mask DB latency.
3. **End of Utterance:** Full stability check runs. If early retrieval matches the final transcript, it is reused.
4. **Decomposition & Retrieval:** The intent is split, retrieved via Hybrid Search + RRF + Cross-Encoder, and merged.
5. **Synthesis:** Streams grounded tokens and citations back to the client UI.
6. **Late Constraints:** If the user speaks a refinement constraint (e.g., "Only AI"), the server intercepts, filters state, and streams a re-grounded answer dynamically.

## 4. Technology Stack
- **Backend:** FastAPI, Python `asyncio`, Uvicorn
- **AI Models:** Gemma 4 12B (Ollama), `sentence-transformers` (PyTorch)
- **Database:** ChromaDB (Dense) + `rank_bm25` (Sparse)
- **UI:** Vanilla JS + CSS, native WebSockets
