# Streaming Live RAG - Edge Cases & Ablations

## Section 1: Analyzed Edge Case Failures

### 1. The "Requery Starvation" Bug (Session Refinement)
**Failure:** The user asked "What is RAG?", receiving a correct synthesis based on 2 chunks from the AI domain. The user then added a late constraint: "Only information about AI". The system output 0 chunks and failed to answer.
**Root Cause:** The `MIN_VIABLE_CHUNKS` starvation threshold was set to 3. Because only 2 chunks existed in state, the delta engine fell back to `REQUERY`. The `REQUERY` logic mistakenly retrieved using a literal appended query ("What is RAG? about ai"), which destroyed the semantic embedding focus, causing the dense retriever to find 0 results.
**Resolution:** `MIN_VIABLE_CHUNKS` was lowered to 1 so `FILTER` mode successfully triggers when >=1 chunk survives. Additionally, the `REQUERY` path was rewritten to retrieve using the *original semantic query* first, and apply the constraint filters deterministically post-retrieval.

### 2. The "Indefinite Streaming Hang" (Synthesizer)
**Failure:** If the local Ollama process crashed (e.g. `CUDA error: shared object initialization failed` or `exit status 0xc0000409`) *during* a streaming response, the FastAPI `async for chunk in response:` loop would hang indefinitely waiting for the next token. The UI spun endlessly.
**Root Cause:** The `asyncio.wait_for` timeout in `core/synthesizer.py` only covered the *initialization* of the LiteLLM response object (Time-to-First-Token). It did not cover the generator's `__anext__` iterations.
**Resolution:** A dedicated `_drain_stream()` async function was introduced and wrapped in a remaining-budget `asyncio.wait_for()`, ensuring the server automatically closes the connection and emits an error if the model stalls mid-stream.

### 3. Substring Collisions in Navigational Suppression
**Failure:** A compound question containing phrases like "What is RAG and how does it reduce hallucinations?" was being misclassified as a `SUPPRESS` navigational command under certain edge cases.
**Root Cause:** The `IntentController` used a simple `if phrase in norm_transcript:` substring check. It was vulnerable to partial matches or overlapping word bounds (e.g. if a phrase was "go to", it would trigger on "I forgot to").
**Resolution:** Updated `NAVIGATIONAL_PHRASES` detection to use strict regex word boundaries `\b` (`re.search(r'\b' + re.escape(phrase) + r'\b', partial.lower())`), eliminating false positive suppression.

---

## Section 2: Ablation Experiments

### Ablation 1: Reciprocal Rank Fusion (RRF) vs. Dense-Only Retrieval
**Hypothesis:** Relying purely on semantic embeddings (ChromaDB + MiniLM) will struggle with exact-keyword queries compared to a Hybrid approach.
**Experiment:** We disabled the `BM25Okapi` sparse branch and the RRF fusion step, passing only the top ChromaDB dense results to the cross-encoder.
**Result:** Semantic-heavy queries ("How does self-attention work?") performed identically. However, acronym-heavy or exact-match queries ("What is JWST?" or "NASA Perseverance") suffered severe drop-offs in Recall@5 because the dense model generalized the acronyms poorly.
**Conclusion:** RRF Fusion of Dense + Sparse is mandatory for domain-specific technical corpuses.

### Ablation 2: Late-Constraint FILTER mode vs Forced REQUERY
**Hypothesis:** Filtering the existing context chunks in-memory is faster and more reliable than executing a fresh database query for late-arriving conversational constraints.
**Experiment:** We temporarily hardcoded the state manager to *always* return `REQUERY`, forcing a full DB round-trip for every refinement (e.g., "...only after 2020"). 
**Result:** 
1. **Latency:** Forced REQUERY added ~300ms to every refinement turn.
2. **Quality:** Applying constraints to a fresh retrieval occasionally surfaced *worse* baseline semantic chunks because the retriever tried to balance semantic similarity with the constraint text, rather than relying on the purely semantic Phase 1 retrieval.
**Conclusion:** The `FILTER` fast-path (applying deterministic metadata filters to the already-retrieved high-quality semantic chunks) is strictly superior for both latency and grounded accuracy, provided `MIN_VIABLE_CHUNKS >= 1` are maintained.
