# Streaming Live RAG - Telemetry Schema

All telemetry events are logged as structured JSON objects to `logs/events.jsonl`. Every event includes an ISO 8601 UTC `ts` (timestamp) and a `session_id`.

## 1. System Events
Used for tracking server lifecycle and corpus initialization.
```json
{
  "ts": "2026-09-29T13:00:00Z",
  "event_type": "system",
  "session_id": "system",
  "extra": {"model": "ollama/gemma4:12b"} // e.g. startup, corpus_loaded
}
```

## 2. Intent Stability
Logged when the controller makes a WAIT / RETRIEVE / SUPPRESS decision at `end_of_utterance`.
```json
{
  "ts": "...",
  "event_type": "stability_decision",
  "session_id": "sess_123",
  "action": "RETRIEVE",
  "confidence": 0.85,
  "reasoning": "rule-based assessment passed",
  "transcript_snippet": "What is RAG?"
}
```

## 3. Query Decomposition
Logged after a stable transcript is decomposed into sub-queries.
```json
{
  "ts": "...",
  "event_type": "decomposed",
  "session_id": "sess_123",
  "turn_id": 1,
  "num_sub_queries": 2,
  "fallback": false
}
```

## 4. Retrieval Phase
Records the output of the Hybrid/RRF pipeline.
```json
{
  "ts": "...",
  "event_type": "retrieval",
  "session_id": "sess_123",
  "sub_queries": [
    {"id": "sq_0", "text": "What is RAG?"}
  ],
  "chunk_counts": {"sq_0": 3},
  "latency_ms": 142.5
}
```

## 5. Synthesis Phase
Records the generator's behavior, latency, and factual grounding footprint.
```json
{
  "ts": "...",
  "event_type": "synthesis",
  "turn_id": "uuid-...",
  "session_id": "sess_123",
  "query": "What is RAG?",
  "action": "RETRIEVE",
  "citations_used": ["[ai_004 §RAG Architecture]", "[ai_010 §Hallucination]"],
  "is_insufficient": false,
  "latency_ms": 1205.2,
  "ttft_ms": 305.1,
  "ttft_mode": "cold",
  "actual_cost_usd": 0.0,
  "list_price_cost_usd": 0.002,
  "sub_query_count": 1,
  "delta_mode": null,
  "prompt_tokens_approx": 12,
  "completion_tokens_approx": 45
}
```

## 6. Session Refinement (Late Constraints)
Logged when a late constraint triggers a delta update.
```json
{
  "ts": "...",
  "event_type": "delta_update",
  "session_id": "sess_123",
  "needs_update": true,
  "new_constraints": {"domain": "AI", "year_min": 2020},
  "delta_mode": "FILTER",
  "augmented_query": null
}
```
