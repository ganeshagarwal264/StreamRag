"""
scripts/measure_metrics.py

Measures real system performance across all scenarios.

Pre-flight sanity check (Fix 2) runs before trusting any recall numbers.
Reports list_price_cost_usd as headline cost metric (Fix 3).

Usage:
    $env:PYTHONPATH = "."; python scripts/measure_metrics.py
"""
import asyncio
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Pre-flight sanity check (Fix 2) ──────────────────────────────────────────

def _cosine(a, b) -> float:
    norm_a = float((a ** 2).sum() ** 0.5) + 1e-9
    norm_b = float((b ** 2).sum() ** 0.5) + 1e-9
    return float(((a / norm_a) * (b / norm_b)).sum())


def run_sanity_preflight() -> bool:
    """
    Hand-written pre-flight check (Fix 2).
    Returns True if both checks pass, False otherwise.
    Recall numbers are UNRELIABLE if this returns False.
    """
    print("=" * 60)
    print("Pre-flight: Embedding + Cross-Encoder Sanity Checks")
    print("=" * 60)

    from core.retriever import _EmbedModel, _CrossEncoder

    model    = _EmbedModel("sentence-transformers/all-MiniLM-L6-v2")
    reranker = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Embedding ordering sanity
    embs = model.encode(
        [
            "What is a neural network?",
            "Neural networks are ML models.",
            "Mars has two moons.",
        ],
        normalize_embeddings=True,
    )
    sim    = _cosine(embs[0], embs[1])
    dissim = _cosine(embs[0], embs[2])
    embed_ok = sim > dissim
    print("  Embedding similar pair:    {:.4f}".format(sim))
    print("  Embedding dissimilar pair: {:.4f}".format(dissim))
    print("  Embedding sanity: {}".format("PASS" if embed_ok else "FAIL -- recall numbers are UNRELIABLE"))

    # Cross-encoder ordering sanity
    q   = "When was the Mars rover launched?"
    rel = "Perseverance rover launched in July 2020 and landed February 2021."
    irr = "Space debris in low Earth orbit poses collision risks to satellites."
    scores = reranker.predict([[q, rel], [q, irr]])
    ce_ok  = float(scores[0]) > float(scores[1])
    print("  Cross-encoder relevant:   {:.4f}".format(float(scores[0])))
    print("  Cross-encoder irrelevant: {:.4f}".format(float(scores[1])))
    print("  Cross-encoder sanity: {}".format("PASS" if ce_ok else "FAIL -- recall numbers are UNRELIABLE"))
    print()

    return embed_ok and ce_ok


# ── Main imports ──────────────────────────────────────────────────────────────

from core.controller import IntentController
from core.decomposer  import IntentDecomposer
from core.retriever   import HybridRetriever
from core.synthesizer import AnswerSynthesizer
from core.state       import state_manager
from telemetry.logger import _list_price

controller = IntentController()
decomposer  = IntentDecomposer()
synthesizer = AnswerSynthesizer()
retriever   = HybridRetriever()

data_dir = Path("data")
with open(data_dir / "transcripts.json", "r", encoding="utf-8") as f:
    TRANSCRIPTS = json.load(f)
with open(data_dir / "corpus.json", "r", encoding="utf-8") as f:
    CORPUS = json.load(f)

# Index corpus so retriever can find documents
retriever.index_corpus(CORPUS)

# Gold doc sets per scenario (heuristic for 40-doc sanity check)
GOLD_DOCS = {
    "sc_01": ["ai_003", "ai_004", "ai_010"],
    "sc_02": ["ai_001", "ai_002", "ai_003", "ai_004", "ai_010"],
    "sc_03": ["cl_005"],
    "sc_06": ["med_001", "med_002", "med_003", "med_004"],
}


async def check_groundedness(query: str, answer: str) -> float:
    """LLM-as-a-judge. Returns 1.0 (grounded) or 0.0 (hallucinated)."""
    if not answer.strip():
        return 0.0
    import litellm
    prompt = (
        "Does the following answer contain only information directly supported "
        "by context, without hallucination?\n"
        "Question: {}\nAnswer: {}\n"
        "Reply ONLY '1' (grounded) or '0' (hallucinated)."
    ).format(query, answer)
    try:
        resp = await litellm.acompletion(
            model="gemini/gemini-3.6-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return 1.0 if "1" in resp.choices[0].message.content.strip() else 0.0
    except Exception:
        return 0.5


async def measure_metrics(sanity_ok: bool):
    print("=" * 60)
    print("System Metrics")
    print("=" * 60)
    if not sanity_ok:
        print("WARNING: Retriever sanity failed -- recall numbers are UNRELIABLE\n")

    total_list_cost   = 0.0
    total_actual_cost = 0.0
    total_ttft        = 0.0
    ttft_n            = 0
    total_ground      = 0.0
    ground_n          = 0
    recall_scores     = []

    for scenario in TRANSCRIPTS:
        sc_id = scenario["scenario_id"]
        print("\n--- [{}] {} ---".format(sc_id, scenario["name"]))

        session_id = "eval_{}".format(sc_id)
        for token in scenario["tokens"]:
            state_manager.append_token(session_id, token)
        session    = state_manager.get_or_create(session_id)
        transcript = session.pending_transcript()
        print("  Transcript: {}".format(transcript))

        decision = await controller.assess(transcript, session)
        match = "OK" if decision.action == scenario["expected_action"] else "MISMATCH"
        print("  Action: {} (expected {}) [{}]".format(
            decision.action, scenario["expected_action"], match))

        if decision.action != "RETRIEVE":
            print("  Skipping retrieval.")
            continue

        sub_queries = await decomposer.decompose(transcript)
        sq_match = "OK" if len(sub_queries) == scenario["expected_sub_queries"] else "~"
        print("  Sub-queries: {} (expected {}) [{}]".format(
            len(sub_queries), scenario["expected_sub_queries"], sq_match))

        # Retrieval
        chunks_by_query   = await retriever.retrieve_multi(sub_queries)
        retrieved_doc_ids = [c.doc_id for cs in chunks_by_query.values() for c in cs]

        # Recall@K
        gold = GOLD_DOCS.get(sc_id)
        if gold:
            hits   = sum(1 for g in gold if g in retrieved_doc_ids)
            recall = hits / len(gold)
            recall_scores.append(recall)
            print("  Recall@K: {:.1f}%  ({}/{} gold chunks found)".format(
                recall * 100, hits, len(gold)))
        else:
            print("  Recall@K: N/A")

        # Synthesis with TTFT timing
        first_token_time = [0.0]
        full_answer      = []
        cost_box         = [0.0]
        start = time.time()

        async def on_token(tok):
            if first_token_time[0] == 0.0:
                first_token_time[0] = time.time()
            full_answer.append(tok)

        async def on_citation(doc_id, section):
            pass

        async def on_done(result, cost_usd=0.0):
            cost_box[0] = cost_usd

        await synthesizer.synthesize_merged(
            sub_queries, chunks_by_query, on_token, on_citation, on_done
        )

        answer = "".join(full_answer)

        # TTFT
        if first_token_time[0] > 0:
            ttft = (first_token_time[0] - start) * 1000
            total_ttft += ttft
            ttft_n     += 1
            print("  TTFT (cold):        {:.1f} ms".format(ttft))

        # Cost: list price vs actual (Fix 3)
        prompt_tokens     = max(1, len(transcript) // 4)
        completion_tokens = max(1, len(answer) // 4)
        list_cost = _list_price(prompt_tokens, completion_tokens)
        actual    = cost_box[0]
        total_list_cost   += list_cost
        total_actual_cost += actual
        print("  List price cost:    ${:.6f}".format(list_cost))
        print("  Actual spend:       ${:.6f}  (free tier)".format(actual))

        # Groundedness
        g = await check_groundedness(transcript, answer)
        total_ground += g
        ground_n     += 1
        print("  Groundedness:       {:.1f}%".format(g * 100))

    # Aggregate
    print("\n" + "=" * 60)
    print("Aggregate Metrics")
    print("=" * 60)
    if ttft_n:
        print("  Avg TTFT (cold):          {:.1f} ms".format(total_ttft / ttft_n))
    if recall_scores:
        avg_recall = sum(recall_scores) / len(recall_scores)
        print("  Avg Recall@K:             {:.1f}%  ({} scenarios)".format(
            avg_recall * 100, len(recall_scores)))
    if ground_n:
        print("  Avg Groundedness:         {:.1f}%".format(
            (total_ground / ground_n) * 100))
    print("  Total list price cost:    ${:.6f}".format(total_list_cost))
    print("  Total actual spend:       ${:.6f}  (free tier -- $0 actual)".format(total_actual_cost))

    from telemetry.logger import logger
    total_turns = len(TRANSCRIPTS)
    llm_calls = logger.stats['stability_llm_fallback'] + logger.stats['decomposer_llm_call'] + len([s for s in TRANSCRIPTS if s['expected_action'] == 'RETRIEVE'])
    avg_llm_calls = llm_calls / total_turns if total_turns else 0
    
    print(f"  Avg LLM calls/turn:       {avg_llm_calls:.2f}  (Total: {llm_calls})")
    print(f"  Stability heuristics:     {logger.stats['stability_heuristic']} pure / {logger.stats['stability_llm_fallback']} LLM")
    print(f"  Decomposer hits:          {logger.stats['decomposer_skipped']} skipped / {logger.stats['decomposer_llm_call']} LLM")

    print()
    print("Note: 40-chunk corpus. Recall is a sanity check for RRF fusion/decomposition,")
    print("not a rigorous academic benchmark (corpus too small for high-variance measurement).")
    if not sanity_ok:
        print("WARNING: Recall figures should be discarded -- retriever sanity check FAILED.")
    print("=" * 60)

    await run_calibration(controller)

async def run_calibration(controller):
    cal_path = Path("data/calibration_set.json")
    if not cal_path.exists():
        print("\nCalibration set not found -- skipping.")
        return

    from core import SessionState
    labeled = json.loads(cal_path.read_text(encoding="utf-8"))
    buckets = {"0.50-0.70": [], "0.70-0.80": [], "0.80-0.90": [], "0.90-1.00": []}

    for item in labeled:
        session = SessionState(session_id="cal_tmp")
        try:
            decision = await controller.assess(item["transcript"], session)
        except Exception:
            continue
        correct = decision.action == item["correct_action"]
        c = decision.confidence
        if   c < 0.70: buckets["0.50-0.70"].append((c, correct))
        elif c < 0.80: buckets["0.70-0.80"].append((c, correct))
        elif c < 0.90: buckets["0.80-0.90"].append((c, correct))
        else:          buckets["0.90-1.00"].append((c, correct))

    print("\n" + "=" * 60)
    print("Confidence Calibration")
    print("  Bucket       n    AvgConf  ActualAcc  Status")
    print("  " + "-" * 50)
    for bucket, entries in buckets.items():
        if not entries:
            print("  [{:<10}]  0    --       --         (no data)".format(bucket))
            continue
        confs, corrects = zip(*entries)
        avg_conf   = sum(confs) / len(confs)
        actual_acc = sum(corrects) / len(corrects)
        gap    = abs(avg_conf - actual_acc)
        status = "OK" if gap <= 0.15 else "POORLY CALIBRATED gap={:.2f}".format(gap)
        print("  [{:<10}]  {:<4} {:<8.3f} {:<10.3f} {}".format(
            bucket, len(entries), avg_conf, actual_acc, status))


if __name__ == "__main__":
    sanity_ok = run_sanity_preflight()
    asyncio.run(measure_metrics(sanity_ok))
    if not sanity_ok:
        sys.exit(1)
