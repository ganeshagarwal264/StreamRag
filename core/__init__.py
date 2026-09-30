"""
core/__init__.py
Shared dataclasses and enumerations used across all core modules.
No external dependencies — import freely from any submodule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional


# ── Stability Decision ────────────────────────────────────────────────────────

@dataclass
class StabilityDecision:
    """
    Output of the Intent Stability Detector in controller.py.

    action:
        WAIT     – transcript is incomplete / trailing; hold off retrieval
        RETRIEVE – stable, semantically complete question; proceed with RAG
        SUPPRESS – navigational / presentation-only command; skip retrieval entirely
    confidence: float in [0.0, 1.0]
    reasoning:  short human-readable explanation (for telemetry / debug)
    """
    action: Literal["WAIT", "RETRIEVE", "SUPPRESS"]
    confidence: float
    reasoning: str


# ── Sub-Query ─────────────────────────────────────────────────────────────────

@dataclass
class SubQuery:
    """
    A single atomic question extracted from a compound utterance.

    id:           unique identifier within a turn, e.g. "sq_0", "sq_1"
    text:         the rewritten, self-contained query string
    intent_type:  'factual' | 'comparative' | 'procedural' | 'clarification'
    is_dependent: True if this sub-query's answer depends on another sub-query
    """
    id: str
    text: str
    intent_type: Literal["factual", "comparative", "procedural", "clarification"]
    is_dependent: bool = False


# ── Retrieved Chunk ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single corpus chunk returned by the retriever after RRF + reranking.

    doc_id:   corpus document identifier, e.g. "ai_001"
    section:  section / heading marker, e.g. "Introduction"
    text:     raw chunk text
    score:    cross-encoder rerank score (higher = more relevant)
    rank:     1-indexed final rank after reranking
    metadata: arbitrary key-value pairs from corpus (domain, year, etc.)
    """
    doc_id: str
    section: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        """Returns citation string in required format: [doc_id §section]"""
        return f"[{self.doc_id} §{self.section}]"


# ── Session State ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    Ephemeral per-session memory. Lives only in-process; cleared on disconnect.

    tokens:           list of transcript tokens received this session
    current_answer:   last fully synthesized answer text
    retrieved_chunks: chunks used for the last answer
    active_sub_queries: sub-queries from the last decomposition
    turn_count:       number of completed retrieve+synthesize cycles
    constraints:      dict of late-arriving constraints (e.g. {"year_min": 2020})
    pending_tokens:   tokens accumulated since last RETRIEVE decision
    """
    session_id: str
    tokens: list[str] = field(default_factory=list)
    current_answer: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    active_sub_queries: list[SubQuery] = field(default_factory=list)
    turn_count: int = 0
    constraints: dict[str, Any] = field(default_factory=dict)
    pending_tokens: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def full_transcript(self) -> str:
        """All accumulated tokens joined as a string."""
        return " ".join(self.tokens)

    def pending_transcript(self) -> str:
        """Tokens accumulated since the last retrieval decision."""
        return " ".join(self.pending_tokens)


# ── Delta Instruction ─────────────────────────────────────────────────────────

@dataclass
class DeltaInstruction:
    """
    Output of the answer delta engine when late constraints arrive.

    needs_update:             True if the current answer must be refined
    new_constraints:          dict of newly detected constraints
    refinement_prompt_suffix: additional instruction appended to the next
                              synthesizer prompt to honour the constraints
    mode:                     "FILTER"  — re-rank existing candidates (cheap path)
                              "REQUERY" — starvation guard: re-run hybrid retrieval
                                          with an augmented, constraint-aware query
    chunks:                   pre-filtered chunks to use in FILTER mode (None = REQUERY)
    augmented_query:          constraint-augmented query string for REQUERY mode
    """
    needs_update: bool
    new_constraints: dict[str, Any]
    refinement_prompt_suffix: str = ""
    mode: Literal["FILTER", "REQUERY"] = "FILTER"
    chunks: Optional[list] = None          # populated in FILTER mode
    augmented_query: Optional[str] = None  # populated in REQUERY mode


# ── WebSocket Event Models ────────────────────────────────────────────────────

@dataclass
class WSEvent:
    """Generic WebSocket event envelope (client → server or server → client)."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.payload}


# ── Synthesis Result ──────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """
    Complete result after a full synthesize cycle, for telemetry/state updates.
    """
    answer: str
    citations: list[str]                   # list of citation_key strings
    is_insufficient: bool = False
    clarifying_question: Optional[str] = None
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
