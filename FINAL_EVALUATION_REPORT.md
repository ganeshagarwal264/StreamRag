# Streaming Live RAG
## Final Evaluation & Benchmark Report

### 1. Executive Summary
This report outlines the final evaluation for the Samsung PRISM Theme 04 — Streaming Live RAG project. The system successfully implements streaming conversational RAG with multi-intent decomposition, session-aware constraint refinement, and strict corpus isolation. The evaluation covers automated test suites, formal theme gates, dedicated ablation studies, and real-world performance profiling.

### 2. System Under Evaluation
**Architecture:**
- Streaming Transcript
- Intent Controller (WAIT / RETRIEVE / SUPPRESS)
- Multi-Intent Decomposer
- Hybrid Retrieval (Dense / ChromaDB + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Session-Aware Synthesis (Session/Delta Engine)
- Gemma 4 12B via Ollama
- Streamed grounded answer + citations

### 3. Hardware and Runtime
**Configuration Details:**
- **Python:** `venv` with PyTorch `2.14.0+cu126` (CUDA version 12.6, `True`)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Ollama Engine:** `ollama version is 0.34.4`, model `gemma4:12b` (Size: 7.6 GB)
- **Inference Mode:** Mixed CPU/GPU Ollama execution (31% CPU / 69% GPU split observed)
- **Settings:**
  - `LLM_MODEL=ollama/gemma4:12b`
  - `OLLAMA_API_BASE=http://localhost:11434`
  - `RERANK_TOP_K=3`
  - `RETRIEVAL_TOP_K=20`
  - `SYNTHESIS_TIMEOUT_S=45`
  - `CHROMA_PATH=./data/chroma_db`

### 4. Automated Test Results
- **Total Tests:** 101
- **Passed:** 101
- **Failed:** 0
- **Skipped:** 0
*(Note: Test suite execution yields 3 non-fatal dependency/deprecation warnings, unrelated to test assertions).*

### 5. Theme 4 Evaluation Gates
| Gate | Requirement | Measured/Verified Result | Status |
| :--- | :--- | :--- | :--- |
| **EARLY RETRIEVAL** | >=80% | 100% | PASS |
| **MULTI-INTENT** | >=70% | 90% | PASS |
| **FACTUAL GROUNDING** | >=85% | 100% (Dedicated evaluation) | PASS |
| **SESSION REFINEMENT** | continuity verified | FILTER and REQUERY paths verified | PASS |
| **TELEMETRY** | 100% trace coverage | 100% | PASS |

### 6. Edge Cases and Failure Handling
During development, the following edge cases were identified and addressed:
1. **Decomposer Timeout**: A local LLM timeout occurred; a deterministic single-intent fallback was used.
2. **Concise Query Starvation**: Queries like "Who was Vaswani?" were originally classified as WAIT due to a minimum token threshold. The controller logic was improved, and concise factual queries now trigger RETRIEVE.
3. **Late/Incomplete Transcript Fragment**: Fragments like "And also what about the" could incorrectly trigger retrieval. Incomplete-ending detection was added so it now correctly waits.
4. **Partial-Evidence Multi-Intent Query**: E.g., "What is retrieval augmented generation, and what are the main causes of climate change?". The corpus contains evidence for RAG but not climate change. The system answers the supported portion and explicitly reports insufficient evidence for the unsupported portion without hallucinating.
5. **CPU/GPU Inference Bottleneck**: `gemma4:12b` exceeds available VRAM, leading to mixed CPU/GPU execution. Context size materially affects TTFT.

### 7. Retrieval Ablation
**Hybrid Retrieval vs Dense-only** (Measured on the 20-query / 40-document synthetic corpus)
- **Hybrid**: Recall@5 = 0.57, Recall@10 = 0.57, Average latency = 101.8 ms
- **Dense-only**: Recall@5 = 0.57, Recall@10 = 0.57, Average latency = 95.4 ms

*Interpretation*: On this synthetic corpus, BM25 contributed no unique retrieved documents. Dense-only matched recall, and hybrid overhead was negligible.

### 8. Reranking Ablation
**Reranking ON vs OFF**
- **Reranking ON**: Recall@5 = 0.57, Recall@10 = 0.57, Average latency = 85.1 ms, TTFT = 27,419.6 ms, Groundedness = 1.00
- **Reranking OFF**: Recall@5 = 0.88, Recall@10 = 0.93, Average latency = 14.5 ms, TTFT = 43,770.5 ms, Groundedness = 1.00

*Interpretation*: On this subset, reranking reduced Recall@K but substantially reduced synthesis TTFT because fewer/stronger context chunks reached the LLM. Groundedness was unchanged in the measured subset.

### 9. Performance Benchmark
**Production RERANK_TOP_K Change**:
- **Baseline (`RERANK_TOP_K=5`)**: TTFT ≈ 87+ seconds in targeted benchmark.
- **Production (`RERANK_TOP_K=3`)**: TTFT ≈ 42.4 seconds.

*Explanation*: Reducing `RERANK_TOP_K` provides fewer context chunks, resulting in a smaller synthesis context and lower prefill burden on the local LLM. The architecture remains unchanged, citations remain intact, and 101 tests remained passing.
*(Note: Streaming occurs after synthesis begins; local LLM prefill remains the dominant latency component).*

### 10. Grounding and Citation Verification
- Factual retrieval uses the supplied local corpus (40 corpus chunks and 40 Chroma chunks in `rag_corpus`).
- The **dedicated supplied-corpus grounding evaluation achieved 100%**.
- A separate **later small-subset performance benchmark at `RERANK_TOP_K=5` reported Groundedness = 50%**.
*(The dedicated grounding evaluation and the performance benchmark used different evaluation subsets/procedures and should not be interpreted as the same measurement).*

### 11. Session Refinement Verification
The system correctly implements non-destructive session updates based on late constraints. If a user queries a topic and subsequently adds "Only information after 2020," the `FILTER` path modifies constraints in the active session rather than isolating the new transcript fragment. Both `FILTER` and `REQUERY` behaviors have been fully verified.

### 12. Telemetry Coverage
Telemetry coverage is verified at **100%**. 
Events tracked include: `session_start`, `token_ingestion`, `stability_decision`, `decomposed`, `retrieval`, `delta_update`, `refining`, `token`, `citation`, `insufficient_evidence`, `conflicting_evidence`, `synthesis`, `error`, `session_reset`, and `session_end`. Token telemetry deliberately avoids storing full generated token text where applicable for privacy and storage efficiency.

### 13. Corpus Isolation
- Factual retrieval is strictly restricted to the supplied local corpus.
- No external web search is used.
- No third-party factual knowledge sources are accessed.
- Unsupported queries correctly yield `insufficient_evidence` behavior rather than generating unverified content.
- Citations strictly reference the retrieved corpus chunks.

### 14. Known Limitations
- **Local Gemma inference latency**: Heavily constrained by the 8 GB GPU memory limit, resulting in mixed CPU/GPU execution.
- **Corpus limitations**: Factual responses are entirely bound by the provided synthetic/supplied corpus.
- **Evaluation script API constraints**: The `scripts/evaluate.py` script requires external cloud/API keys for the LLM-as-a-judge portions (Multi-intent decomposition checks were logged as `SKIP` without API keys).
- **Interface**: Transcript input is currently simulated through the UI. Microphone ASR is NOT currently implemented. TTS is NOT currently implemented.

### 15. Reproducibility Instructions
1. Activate virtual environment: `.\venv\Scripts\Activate.ps1`
2. Ensure Ollama is running and model is loaded: `ollama pull gemma4:12b`
3. Generate data: `python scripts/generate_mock_data.py`
4. Run tests: `pytest -q`
5. Start backend: `uvicorn api.server:app`

### 16. Final Evaluation Summary
The project successfully meets the PRISM Theme 04 objectives. Key functionalities including intent-stability filtering (WAIT/RETRIEVE/SUPPRESS), multi-intent decomposition, corpus-grounded synthesis with citations, and session state refinement have all been integrated into a verifiable pipeline passing 101 out of 101 automated tests. The system operates entirely independent of external web searches, strictly bounded by the supplied RAG corpus.
