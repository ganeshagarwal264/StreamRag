"""
telemetry/logger.py

Structured event logger for the Streaming Live RAG system.

Every synthesis event records both:
  actual_cost_usd   â€” true spend (always $0 on Gemini free tier)
  list_price_cost_usd â€” what the same tokens would cost at published list prices

This makes the metric useful for comparing turn complexity even when the
actual bill is $0 (e.g. REQUERY turns show ~2Ã— list price vs FILTER turns).
"""
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from core import StabilityDecision, SynthesisResult, DeltaInstruction

load_dotenv()

LOG_DIR  = os.environ.get("LOG_DIR",  "./logs")
LOG_FILE = os.environ.get("LOG_FILE", "events.jsonl")

# Published list price model name for litellm.completion_cost
_LIST_PRICE_MODEL = "gemini/gemini-3.6-flash"


def _list_price(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate list-rate cost regardless of whether the account is on the free
    tier.  Returns 0.0 on any error (e.g. model not in litellm cost map).
    """
    try:
        import litellm
        return litellm.completion_cost(
            completion_response={
                "model": _LIST_PRICE_MODEL,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
    except Exception:
        return 0.0


class TelemetryLogger:
    def __init__(self):
        self._log_path = Path(LOG_DIR) / LOG_FILE
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'stability_llm_fallback': 0,
            'stability_heuristic': 0,
            'decomposer_llm_call': 0,
            'decomposer_skipped': 0,
        }

    def _write(self, event: dict) -> None:
        event["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    # â”€â”€ Stability decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def log_stability_decision(
        self,
        session_id: str,
        transcript: str,
        decision: StabilityDecision,
    ) -> None:
        is_llm = decision.reasoning in ('parsed from llm', 'llm check failed')
        if is_llm:
            self.stats['stability_llm_fallback'] += 1
        else:
            self.stats['stability_heuristic'] += 1
        
        self._write({
            "event_type":        "stability_decision",
            "session_id":        session_id,
            "action":            decision.action,
            "confidence":        decision.confidence,
            "reasoning":         decision.reasoning,
            "transcript_snippet": transcript[-100:] if transcript else "",
        })

    # â”€â”€ Retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def log_retrieval(
        self,
        session_id: str,
        sub_queries: list,
        chunks_by_query: dict,
        latency_ms: float,
    ) -> None:
        self._write({
            "event_type":   "retrieval",
            "session_id":   session_id,
            "sub_queries":  [{"id": sq.id, "text": sq.text} for sq in sub_queries],
            "chunk_counts": {sq_id: len(chunks) for sq_id, chunks in chunks_by_query.items()},
            "latency_ms":   latency_ms,
        })

    # â”€â”€ Synthesis (Fix 3: full structured schema) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def log_synthesis(
        self,
        session_id: str,
        query: str,
        result: SynthesisResult,
        actual_cost_usd: float = 0.0,
        sub_query_count: int = 1,
        delta_mode: str | None = None,
        action: str = "RETRIEVE",
    ) -> None:
        """
        Full synthesis telemetry event.

        actual_cost_usd       True spend (always $0 on Gemini free tier).
        list_price_cost_usd   What the same call would cost at published list
                              price â€” useful for comparing turn complexity.
        ttft_mode             "cold" on first response; "speculative_hit" if
                              retrieval was skipped (FILTER delta mode).
        delta_mode            "FILTER" | "REQUERY" | None
        """
        # Approximate token counts from text length (4 chars â‰ˆ 1 token)
        prompt_tokens     = max(1, len(query) // 4)
        completion_tokens = max(1, len(result.answer) // 4)
        lp = _list_price(prompt_tokens, completion_tokens)

        ttft_mode = (
            "speculative_hit" if delta_mode == "FILTER"
            else "cold"
        )

        self._write({
            "event_type":          "synthesis",
            "turn_id":             str(uuid.uuid4()),
            "session_id":          session_id,
            "query":               query,
            "action":              action,
            "citations_used":      result.citations,
            "is_insufficient":     result.is_insufficient,
            # Latency
            "latency_ms":          result.latency_ms,
            "ttft_ms":             result.ttft_ms,
            "ttft_mode":           ttft_mode,
            # Cost
            "actual_cost_usd":     actual_cost_usd,
            "list_price_cost_usd": lp,
            # Complexity indicators
            "sub_query_count":     sub_query_count,
            "delta_mode":          delta_mode,
            "prompt_tokens_approx":     prompt_tokens,
            "completion_tokens_approx": completion_tokens,
        })

    # â”€â”€ Delta update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def log_delta_update(self, session_id: str, delta: DeltaInstruction) -> None:
        self._write({
            "event_type":      "delta_update",
            "session_id":      session_id,
            "needs_update":    delta.needs_update,
            "new_constraints": delta.new_constraints,
            "delta_mode":      delta.mode,
            "augmented_query": delta.augmented_query,
        })

    def log_token_ingestion(self, session_id: str, turn_id: int, token_count: int) -> None:
        self._write({
            "event_type": "token_ingestion",
            "session_id": session_id,
            "turn_id": turn_id,
            "token_count": token_count,
        })

    def log_decomposed(self, session_id: str, turn_id: int, num_sub_queries: int, fallback: bool) -> None:
        self._write({
            "event_type": "decomposed",
            "session_id": session_id,
            "turn_id": turn_id,
            "num_sub_queries": num_sub_queries,
            "fallback": fallback,
        })

    def log_refining(self, session_id: str, turn_id: int, mode: str) -> None:
        self._write({
            "event_type": "refining",
            "session_id": session_id,
            "turn_id": turn_id,
            "mode": mode,
        })

    def log_token(self, session_id: str, turn_id: int, token_index: int) -> None:
        self._write({
            "event_type": "token",
            "session_id": session_id,
            "turn_id": turn_id,
            "token_index": token_index,
        })

    def log_citation(self, session_id: str, turn_id: int, doc_id: str, section: str) -> None:
        self._write({
            "event_type": "citation",
            "session_id": session_id,
            "turn_id": turn_id,
            "doc_id": doc_id,
            "section": section,
        })

    def log_insufficient_evidence(self, session_id: str, turn_id: int) -> None:
        self._write({
            "event_type": "insufficient_evidence",
            "session_id": session_id,
            "turn_id": turn_id,
        })

    def log_conflicting_evidence(self, session_id: str, turn_id: int, doc_ids: list[str]) -> None:
        self._write({
            "event_type": "conflicting_evidence",
            "session_id": session_id,
            "turn_id": turn_id,
            "doc_ids": doc_ids,
        })

    def log_error(self, session_id: str, turn_id: int, error_msg: str, recoverable: bool, error_type: str = "") -> None:
        self._write({
            "event_type": "error",
            "session_id": session_id,
            "turn_id": turn_id,
            "error_msg": error_msg,
            "recoverable": recoverable,
            "error_type": error_type,
        })

    # â”€â”€ Generic session events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def log_session_event(
        self,
        session_id: str,
        event_type: str,
        extra: dict = None,
    ) -> None:
        event = {"event_type": event_type, "session_id": session_id}
        if extra:
            event.update(extra)
        self._write(event)


logger = TelemetryLogger()
