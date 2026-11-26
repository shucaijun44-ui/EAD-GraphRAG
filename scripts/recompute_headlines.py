"""Recompute headline point estimates from public aggregate counts."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = {}
    with (ROOT / "results/headline_rank_counts.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            counts = {key: int(row[key]) for key in ("rank1", "rank2", "rank3", "miss_or_failure")}
            assert sum(counts.values()) == n
            results[row["cohort"]] = {
                "n": n, "Acc@1": counts["rank1"] / n,
                "Recall@3": sum(counts[f"rank{k}"] for k in (1, 2, 3)) / n,
                "MRR@3": sum(counts[f"rank{k}"] / k for k in (1, 2, 3)) / n,
            }
    with (ROOT / "results/T09_PrimaryD1_283.csv").open(encoding="utf-8") as handle:
        decision = next(row for row in csv.DictReader(handle) if row["policy"] == "D1_support2")
    n = int(decision["n"])
    accepted = int(decision["accepted_n"])
    correct = int(decision["correct_accepted_n"])
    assert accepted == correct + int(decision["incorrect_accepted_n"])
    results["mixed_task_D1"] = {"n": n, "accepted_n": accepted,
        "Coverage": accepted / n, "Accepted_Accuracy": correct / accepted,
        "Withholding_Sensitivity": float(decision["Sensitivity"]),
        "Diagnose_Specificity": float(decision["Specificity"])}
    results["scope"] = "aggregate point-estimate verification; no model rerun or confidence-interval recomputation"
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
