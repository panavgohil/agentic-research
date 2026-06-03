"""
Evaluation runner for all ablation configurations.

Usage:
    python run_eval.py                     # Run all configs
    python run_eval.py --config full_agent # Run one config
    python run_eval.py --score             # Score existing predictions
"""

import os
import json
import argparse
import time
import re
from pathlib import Path

# ── Configuration registry ────────────────────────────────────────────────────
CONFIGS = {
    "full_agent": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid": True,
        "baseline": False,
    },
    "baseline": {
        "use_planner": False,
        "use_reflector": False,
        "use_citation_verifier": False,
        "use_hybrid": True,
        "baseline": True,
    },
    "no_planner": {
        "use_planner": False,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid": True,
        "baseline": False,
    },
    "no_reflector": {
        "use_planner": True,
        "use_reflector": False,
        "use_citation_verifier": True,
        "use_hybrid": True,
        "baseline": False,
    },
    "no_citation_verifier": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": False,
        "use_hybrid": True,
        "baseline": False,
    },
    "no_hybrid": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid": False,
        "baseline": False,
    },
}

BASE_DIR = Path(__file__).parent
EVAL_PATH = BASE_DIR / "eval" / "questions.jsonl"
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def load_questions() -> list[dict]:
    questions = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_config(config_name: str, questions: list[dict], resume: bool = True) -> None:
    from agent.agent import run as agent_run

    cfg = CONFIGS[config_name]
    out_path = PRED_DIR / f"{config_name}.jsonl"

    # Resume: skip already-answered questions
    done_ids = set()
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done_ids.add(entry["id"])
                except Exception:
                    pass
        print(f"  Resuming {config_name}: {len(done_ids)}/30 already done")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for q in questions:
            if q["id"] in done_ids:
                continue
            print(f"  [{config_name}] {q['id']}: {q['question'][:60]}...")
            try:
                result = agent_run(question=q["question"], **cfg)
                entry = {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "trace": result["trace"],
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                entry = {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": "",
                    "citations": [],
                    "trace": {
                        "subquestions": [],
                        "retrieval_rounds": 0,
                        "tool_calls": 0,
                        "reflection_decision": "error",
                        "latency_seconds": 0.0,
                        "error": str(e),
                    },
                }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            # Be kind to rate limits
            time.sleep(1.5)


# ── Scoring ───────────────────────────────────────────────────────────────────

def citation_metrics(predicted: list[str], must_cite: list[str]) -> dict:
    pred_set = set(predicted)
    must_set = set(must_cite)
    tp = pred_set & must_set
    precision = len(tp) / len(pred_set) if pred_set else 0.0
    recall = len(tp) / len(must_set) if must_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def score_all() -> None:
    """Compute citation metrics for all configs (no LLM judge needed)."""
    questions = load_questions()
    must_cite_map = {q["id"]: q.get("must_cite", []) for q in questions}

    print("\n" + "=" * 80)
    print(f"{'Config':<25} {'Cit-P':>7} {'Cit-R':>7} {'Cit-F1':>7} {'Avg-Tools':>10} {'Avg-Latency':>12}")
    print("-" * 80)

    rows = []
    for config_name in CONFIGS:
        pred_path = PRED_DIR / f"{config_name}.jsonl"
        if not pred_path.exists():
            print(f"  {config_name:<23} (no predictions file)")
            continue

        preds = []
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                try:
                    preds.append(json.loads(line))
                except Exception:
                    pass

        if not preds:
            continue

        precisions, recalls, f1s = [], [], []
        tool_counts, latencies = [], []

        for pred in preds:
            qid = pred["id"]
            must = must_cite_map.get(qid, [])
            metrics = citation_metrics(pred.get("citations", []), must)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])
            trace = pred.get("trace", {})
            tool_counts.append(trace.get("tool_calls", 0))
            latencies.append(trace.get("latency_seconds", 0))

        row = {
            "config": config_name,
            "n": len(preds),
            "cit_p": round(sum(precisions) / len(precisions), 3),
            "cit_r": round(sum(recalls) / len(recalls), 3),
            "cit_f1": round(sum(f1s) / len(f1s), 3),
            "avg_tools": round(sum(tool_counts) / len(tool_counts), 1),
            "avg_latency": round(sum(latencies) / len(latencies), 1),
        }
        rows.append(row)
        print(
            f"  {config_name:<23} {row['cit_p']:>7.3f} {row['cit_r']:>7.3f}"
            f" {row['cit_f1']:>7.3f} {row['avg_tools']:>10.1f} {row['avg_latency']:>11.1f}s"
            f"  (n={row['n']})"
        )

    # Save scores
    scores_path = BASE_DIR / "eval" / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nScores saved to {scores_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default=None,
                        help="Run a single config (default: all)")
    parser.add_argument("--score", action="store_true",
                        help="Score existing predictions only (no inference)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run from scratch, overwrite existing predictions")
    args = parser.parse_args()

    if args.score:
        score_all()
    else:
        questions = load_questions()
        configs_to_run = [args.config] if args.config else list(CONFIGS.keys())
        for cfg in configs_to_run:
            print(f"\n>>> Running config: {cfg}")
            run_config(cfg, questions, resume=not args.no_resume)
        print("\n>>> Scoring...")
        score_all()
