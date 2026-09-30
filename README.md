# Streaming Live RAG System

An end-to-end prototype system for incremental retrieval, multi-intent decomposition, and session-aware answer refinement, built around a local Gemma 4 12B inference engine via Ollama.

## 1. Problem and Theme
This system addresses streaming conversational RAG where a natural utterance may contain multiple intents and where late-arriving constraints should refine an existing answer. 

## 2. Architecture
```text
Streaming Transcript
        ↓
Intent Controller
        ↓
WAIT / RETRIEVE / SUPPRESS
        ↓
Multi-Intent Decomposer
        ↓
Hybrid Retrieval
    ├── Dense / ChromaDB
    └── BM25
        ↓
Reciprocal Rank Fusion
        ↓
Cross-Encoder Reranking
        ↓
Session-Aware Synthesis
        ↓
Gemma 4 12B via Ollama
        ↓
Streamed grounded answer + citations
```

### Core Components
- **Intent Controller**: Analyzes transcript tokens and decides if the system should WAIT, RETRIEVE, or SUPPRESS.
- **Multi-Intent Decomposer**: Breaks compound questions into parallel sub-queries to handle multi-intent inputs.
- **Hybrid Retriever**: Combines BM25 and ChromaDB vector search.
- **RRF (Reciprocal Rank Fusion)**: Fuses results from the dense and sparse retrievers.
- **Cross-Encoder Reranking**: Reranks the fused candidates for maximum relevance.
- **Session/Delta Engine**: Detects late-arriving constraints (e.g., "after 2020") and refines answers mid-session.
- **Session-Aware Synthesizer**: Streams grounded responses backed by corpus citations.
- **Grounded Citations**: Formats evidence sources and outputs strict inline citations.
- **Telemetry**: Records key metrics and decision tree paths.

## 3. Current Runtime
- **Python Environment**: `venv`
- **PyTorch**: `2.14.0+cu126` (CUDA available: True, CUDA version: 12.6)
- **Ollama Engine**: Local execution via Ollama
- **Model**: `gemma4:12b`
- **Endpoint**: `http://localhost:11434`
- **Hardware Profile**: RTX 4060 CUDA-enabled PyTorch environment
- **Inference Mode**: Mixed CPU/GPU Ollama execution (Observed: 31% CPU / 69% GPU split for Gemma 4 12B)
- **Configuration Defaults**:
  - `RERANK_TOP_K=3`
  - `RETRIEVAL_TOP_K=20`
  - `SYNTHESIS_TIMEOUT_S=45`

## 4. Prerequisites
- Python 3.12+
- Ollama
- `gemma4:12b` (Available locally)
- NVIDIA GPU is supported by the current tested environment (but not a hard requirement to launch, assuming sufficient system RAM).

## 5. Setup on Windows / PowerShell
Open a terminal and set up the Python environment:
```powershell
cd "C:\Users\lucky\OneDrive\Desktop\prism hackathon"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify and prepare Ollama:
```powershell
ollama --version
ollama pull gemma4:12b
```
*(If Ollama is already running as a Windows service, do not start a second server. Use `ollama ps` to inspect loaded models).*

## 6. Run Tests
Execute the automated test suite:
```powershell
pytest -q
```
**Current verified result:** 101 passed, 0 failed, 0 skipped.
*(Note: The test run currently produces 3 non-fatal dependency/deprecation warnings).*

## 7. Run the Application
From the activated `venv`, launch the backend:
```powershell
uvicorn api.server:app --reload
```
Open the interface in your browser:
http://localhost:8000

## 8. WebSocket Protocol
The system uses the following JSON event schema:

**Client → Server:**
- `token`: Streams ongoing speech fragments
- `end_of_utterance`: Triggers intent processing
- `reset`: Clears the session

**Server → Client:**
- `status`: Emits WAIT, RETRIEVE, or SUPPRESS decisions
- `decomposed`: Emits atomic sub-queries for multi-intent inputs
- `retrieved`: Signals completed vector retrieval
- `token`: Steamed generation token from LLM
- `citation`: Grounded reference identification
- `done`: Synthesis complete
- `delta_detected`: Late constraint found
- `refining`: Acknowledges re-synthesis action
- `insufficient_evidence`: Warns when the corpus lacks data
- `conflicting_evidence`: Warns if retrieved evidence contradicts itself
- `error`: Signals a backend exception
- `reset_ack`: Confirms session reset

## 9. Corpus Isolation
- Factual retrieval uses the strictly supplied local corpus.
- No external web search or general web-knowledge is used.
- Unsupported evidence queries trigger explicit `insufficient_evidence` behavior (no hallucination).
- Citations strictly identify precise corpus chunks/sections.

## 10. Session-Aware Refinement
The system supports non-destructive refinement (FILTER vs REQUERY).

**Turn 1:**
"What is retrieval augmented generation?"
*(System responds)*

**Turn 2:**
"Only information after 2020."

The system detects the late constraint (FILTER), retains the prior context, and automatically refines the existing session constraints to fetch appropriately filtered data rather than treating the instruction as a completely isolated search.

## 11. Intent Controller
The controller intercepts speech and directs flow:
- **WAIT**: "um uh well so"
- **RETRIEVE**: "What is retrieval augmented generation?"
- **SUPPRESS**: "next slide please"

*(These are examples; the LLM/heuristic engine dynamically parses varied inputs).*

## 12. Multi-Intent Handling
**Example:**
"What is retrieval augmented generation, and what are the main causes of climate change?"

The engine automatically produces parallel sub-queries (e.g., `sq_0`: RAG, `sq_1`: Climate Change). Any unsupported sub-queries receive `insufficient_evidence` handling independently of the successful factual retrievals.

## 13. Grounding and Citations
Answers are anchored by specific corpus records.
Example citation format: `[ai_003 §RAG Overview]`
*(Note: While highly constrained, no generative system guarantees absolute perfect universal grounding).*

## 14. Telemetry
Structured telemetry records major pipeline events for audit and performance.
Event types include: `stability_decision`, `retrieval`, `synthesis`, `delta_update`, `token_ingestion`, `decomposed`, `refining`, `token`, `citation`, `insufficient_evidence`, `conflicting_evidence`, `error`, and `session_event`.

## 15. Performance
Measurements are heavily reliant on available hardware context. Local Gemma 4 12B inference on an 8 GB RTX 4060 utilizes mixed CPU/GPU execution (e.g., 31% CPU / 69% GPU).
Latency is context-dependent, and the system is not universally real-time. TTFT (Time-to-First-Token) heavily scales by hardware capabilities and the `RERANK_TOP_K` context size.

## 16. Evaluation Status
- **101/101 automated tests passing**
- early retrieval gate status: Verified
- multi-intent gate status: Verified
- factual grounding evaluation: Completed (Note: Subset benchmarking reported lower experimental metrics due to prompt limitations, contrasting the dedicated 100% test evaluation).
- session refinement verification: Verified
- telemetry coverage: Implemented
- ablation experiments: Recorded

## 17. Known Limitations
- Local Gemma inference latency is bottlenecked by hardware.
- The RTX 4060's 8 GB GPU memory constraint forces CPU spillover for a 12B model.
- Factual knowledge is bounded completely by the synthetic/supplied corpus limitations.
- Transcript input is currently simulated strictly through the text UI.
- Microphone ASR is **NOT** currently implemented.
- Text-to-Speech (TTS) is **NOT** currently implemented.

## 18. Troubleshooting
- **Ollama not running**: Launch `ollama serve` or start the background service.
- **Model unavailable**: `ollama pull gemma4:12b`
- **Port 8000 already in use**: Stop any existing `uvicorn` instances.
- **Python environment not activated**: Run `.\venv\Scripts\Activate.ps1`.
- **CUDA Verification**: `python -c "import torch; print(torch.cuda.is_available())"`
- **Chroma/corpus mismatch**: Run `python scripts/generate_mock_data.py` to rebuild local data.

## 19. Project Status
Core Streaming Live RAG prototype:
- Working
- 101 automated tests passing
- CUDA-enabled project environment verified
- Gemma 4 12B via Ollama verified
- 40 corpus chunks / 40 Chroma chunks
- WAIT / RETRIEVE / SUPPRESS verified
- multi-intent verified
- session refinement verified
- citations verified
- telemetry implemented

**Voice microphone input and text-to-speech are NOT yet implemented.**
*(Note: This project is not submission-complete. Architecture brief, final evaluation package, demo recording, and voice enhancements are still planned).*
