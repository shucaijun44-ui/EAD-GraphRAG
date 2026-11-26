"""Evaluate separate local gold and prediction files with the archived scorer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ead_graphrag import evaluation
from ead_graphrag.runtime import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    if args.output.exists():
        raise FileExistsError("Choose a new output filename")
    gold = load_rows(args.gold)
    if not gold:
        raise ValueError("Gold cohort must not be empty")
    if len({row["question_id"] for row in gold}) != len(gold):
        raise ValueError("Duplicate gold identifiers")
    pred = load_rows(args.predictions)
    rows = evaluation.score(gold, pred)
    intervals = evaluation.summarize("user_provided", "user_prediction", rows,
        args.bootstrap_replicates, args.seed, "local evaluation; resources not distributed")
    lookup = {row["question_id"]: row for row in pred}
    positive = [row for row in gold if row.get("requires_clarification") or row.get("subset") in ("ambiguous", "ood_unanswerable")]
    negative = [row for row in gold if row not in positive]

    def valid(g):
        p = lookup.get(g["question_id"])
        return p is not None and not p.get("error")

    accepted = [row for row in gold if valid(row) and lookup[row["question_id"]].get("uncertain") is False]
    correct = sum(bool(lookup[g["question_id"]].get("candidate_diseases")) and
                  lookup[g["question_id"]]["candidate_diseases"][0] in g.get("gold_diseases", [])
                  and g not in positive for g in accepted)
    sensitivity = sum(valid(g) and lookup[g["question_id"]].get("uncertain") is True for g in positive) / len(positive) if positive else None
    specificity = sum(valid(g) and lookup[g["question_id"]].get("uncertain") is False for g in negative) / len(negative) if negative else None
    result = {"n": len(gold), "failures_or_missing": sum(not valid(g) for g in gold),
              "ranking_and_retrieval": intervals,
              "decisions": {"Withholding_Sensitivity": sensitivity, "Diagnose_Specificity": specificity,
                "Balanced_Accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
                "accepted_n": len(accepted), "correct_accepted_n": correct,
                "Coverage": len(accepted) / len(gold),
                "Accepted_Accuracy": correct / len(accepted) if accepted else None},
              "note": "Decision summaries are point estimates. Cluster CIs are produced only for the archived scorer endpoints."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(gold), "output": str(args.output)}))


if __name__ == "__main__":
    main()
