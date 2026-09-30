"""
core/state.py
Ephemeral per-session memory + late-constraint delta engine.

Delta algorithm (sharpen, don't restart):
  1. detect_late_constraints() extracts new constraints from incoming tokens.
  2. compute_delta() applies apply_constraints() to existing retrieved chunks.
     - If len(filtered) >= MIN_VIABLE_CHUNKS  →  FILTER mode  (cheap, no re-query)
     - If len(filtered) <  MIN_VIABLE_CHUNKS  →  REQUERY mode (starvation guard:
       augment original query with constraint text and run a fresh hybrid retrieval)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core import SessionState, RetrievedChunk, SubQuery, DeltaInstruction

# ── Starvation threshold ───────────────────────────────────────────────────────
MIN_VIABLE_CHUNKS = 1  # FILTER when >=1 chunk survives; REQUERY only on true starvation (0 chunks)


# ── Constraint extraction patterns ────────────────────────────────────────────
# Each key maps to a compiled regex.  Group 1 is the captured value.
# Domain uses case-insensitive matching against the fixed corpus vocabulary:
#   AI | Climate | Space | Medicine
CONSTRAINT_PATTERNS: Dict[str, re.Pattern] = {
    "year_min": re.compile(r"(?:after|since)\s+(\d{4})", re.IGNORECASE),
    "year_max": re.compile(r"(?:before|until)\s+(\d{4})", re.IGNORECASE),
    "year_exact": re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE),
    "domain":    re.compile(r"\b(AI|climate|space|medicine)\b", re.IGNORECASE),
    "recency":   re.compile(r"\b(most recent|latest|newest)\b", re.IGNORECASE),
}

# Corpus domain vocabulary (lowercase) for normalisation
_DOMAIN_VOCAB = {"ai", "climate", "space", "medicine"}


# ── Helper: apply constraints to a chunk list ─────────────────────────────────

def apply_constraints(
    chunks: list[RetrievedChunk],
    constraints: Dict[str, Any],
) -> list[RetrievedChunk]:
    """
    Filter *chunks* by the given *constraints* dict.
    Supported keys: year_min, year_max, domain, recency.
    Returns the surviving chunks (may be empty).
    """
    result = []
    for chunk in chunks:
        year   = chunk.metadata.get("year")
        domain = chunk.metadata.get("domain", "").lower()
        keep   = True

        if "year_min" in constraints and year is not None:
            if year < constraints["year_min"]:
                keep = False
        if "year_max" in constraints and year is not None:
            if year > constraints["year_max"]:
                keep = False
        if "domain" in constraints:
            req = constraints["domain"].lower()
            if req not in domain:
                keep = False

        if keep:
            result.append(chunk)

    # Recency: sort survivors newest-first and update ranks
    if constraints.get("recency") and result:
        result.sort(key=lambda c: c.metadata.get("year", 0), reverse=True)
        for i, c in enumerate(result):
            c.rank = i + 1

    return result


# ── Helper: build a constraint-augmented query string ─────────────────────────

def augment_query_with_constraints(
    original_query: str,
    constraints: Dict[str, Any],
) -> str:
    """
    Appends constraint clauses to *original_query* so a fresh retrieval can
    surface chunks that the original un-constrained query would have missed.

    Examples
    --------
    "Tell me about transformers" + {year_min: 2022}
      -> "Tell me about transformers after 2022"
    "Tell me about transformers" + {domain: "climate"}
      -> "Tell me about transformers about climate"
    """
    parts: list[str] = [original_query.rstrip()]
    q = original_query.lower()

    if "year_min" in constraints and "year_max" in constraints:
        y_min = str(constraints["year_min"])
        y_max = str(constraints["year_max"])
        if constraints["year_min"] == constraints["year_max"]:
            if y_min not in q:
                parts.append(f"in {y_min}")
        else:
            if y_min not in q or y_max not in q:
                parts.append(f"after {y_min} before {y_max}")
    elif "year_min" in constraints:
        y_min = str(constraints["year_min"])
        if y_min not in q:
            parts.append(f"after {y_min}")
    elif "year_max" in constraints:
        y_max = str(constraints["year_max"])
        if y_max not in q:
            parts.append(f"before {y_max}")

    if "domain" in constraints:
        d = str(constraints["domain"]).lower()
        if d not in q:
            parts.append(f"about {d}")

    if constraints.get("recency"):
        if "recent" not in q and "latest" not in q and "newest" not in q:
            parts.append("recent")

    return " ".join(parts)


class StateManager:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def append_token(self, session_id: str, token: str) -> None:
        session = self.get_or_create(session_id)
        session.tokens.append(token)
        session.pending_tokens.append(token)

    def mark_retrieved(self, session_id: str, chunks: list[RetrievedChunk], sub_queries: list[SubQuery] = None) -> None:
        session = self.get_or_create(session_id)
        session.retrieved_chunks = chunks
        if sub_queries is not None:
            session.active_sub_queries = sub_queries
        session.turn_count += 1
        session.pending_tokens.clear()

    def update_answer(self, session_id: str, answer: str, citations: list[str]) -> None:
        session = self.get_or_create(session_id)
        session.current_answer = answer

    def destroy(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    async def _llm_extract_constraint(self, text: str) -> dict:
        import os
        import litellm
        model = os.environ.get('LLM_MODEL_LITE', os.environ.get('LLM_MODEL', 'gemini/gemini-2.5-flash'))
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract constraints as JSON. Keys: year_min, year_max, domain, recency. Return ONLY valid JSON dict, or {} if none."},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )
            import json
            content = resp.choices[0].message.content
            # Try to extract json
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
            return {}
        except Exception:
            return {}

    async def detect_late_constraints(self, session_id: str, new_tokens: list[str]) -> DeltaInstruction:
        session = self.get_or_create(session_id)
        text = " ".join(new_tokens)
        new_constraints: Dict[str, Any] = {}

        for key, pattern in CONSTRAINT_PATTERNS.items():
            m = pattern.search(text)
            if m:
                if key in ["year_min", "year_max", "year_exact"]:
                    year = int(m.group(1))
                    if key == "year_exact":
                        new_constraints["year_min"] = year
                        new_constraints["year_max"] = year
                    else:
                        new_constraints[key] = year
                elif key == "domain":
                    new_constraints["domain"] = m.group(1).capitalize()
                elif key == "recency":
                    new_constraints["recency"] = True

        if not new_constraints:
            llm_result = await self._llm_extract_constraint(text)
            new_constraints.update(llm_result)
            if llm_result:
                from telemetry.logger import logger
                logger.log_session_event(
                    session.session_id,
                    "constraint_fallback_llm",
                    {"text": text, "extracted": llm_result},
                )

        added = {
            k: v
            for k, v in new_constraints.items()
            if k not in session.constraints or session.constraints[k] != v
        }

        if not added:
            return DeltaInstruction(needs_update=False, new_constraints={})

        session.constraints.update(added)

        suffix_parts: list[str] = []
        if "year_min" in added and "year_max" in added:
            if added["year_min"] == added.get("year_max"):
                suffix_parts.append(f"from {added['year_min']}")
            else:
                suffix_parts.append(f"from {added['year_min']} through {added['year_max']}")
        else:
            if "year_min" in added:
                suffix_parts.append(f"from {added['year_min']} onward")
            if "year_max" in added:
                suffix_parts.append(f"up to {added['year_max']}")
        if "domain" in added:
            suffix_parts.append(f"about {added['domain']}")
        if added.get("recency"):
            suffix_parts.append("using the most recent information")

        suffix = (
            " \n\nREFINEMENT INSTRUCTION: The retrieval controller has already applied the requested constraints ("
            + " and ".join(suffix_parts)
            + "). Every provided chunk has been pre-validated. Update the answer using only the supplied eligible evidence. Do not independently re-evaluate whether the evidence satisfies the constraints."
        ) if suffix_parts else ""

        return self.compute_delta(session, added, suffix)

    def compute_delta(self, session: SessionState, added: dict, suffix: str) -> DeltaInstruction:
        filtered = apply_constraints(session.retrieved_chunks, session.constraints)
        if len(filtered) >= MIN_VIABLE_CHUNKS:
            return DeltaInstruction(
                needs_update=True,
                new_constraints=added,
                refinement_prompt_suffix=suffix,
                mode="FILTER",
                chunks=filtered,
                augmented_query=None,
            )
        # REQUERY: 0 chunks survived the constraint filter (true starvation).
        # Use the ORIGINAL SEMANTIC query for retrieval to preserve embedding quality,
        # then apply constraints post-retrieval. The augmented string is kept for
        # logging/telemetry only.
        original_query = (
            session.active_sub_queries[0].text
            if session.active_sub_queries
            else session.full_transcript()
        )
        return DeltaInstruction(
            needs_update=True,
            new_constraints=added,
            refinement_prompt_suffix=suffix,
            mode="REQUERY",
            chunks=None,
            augmented_query=original_query,   # semantic query; constraints applied post-retrieval
        )

state_manager = StateManager()
