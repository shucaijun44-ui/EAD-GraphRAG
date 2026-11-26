"""Exercise the selected original functions with eight fictional cases."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ead_graphrag.offline import predict_fixture


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo_predictions.jsonl")
    args = parser.parse_args()
    data = ROOT / "data/synthetic"
    cases = read_jsonl(data / "cases.jsonl")
    triples = read_jsonl(data / "graph.jsonl")
    chunks = read_jsonl(data / "chunks.jsonl")
    results = []
    for case in cases:
        inputs = {key: case[key] for key in (
            "question", "crop", "linked_entities", "entity_scores", "fixture_crossencoder_scores"
        )}
        with redirect_stdout(io.StringIO()):
            result = predict_fixture(inputs, triples, chunks)
        result["question_id"] = case["question_id"]
        assert result["uncertain"] == case["requires_clarification"], case["question_id"]
        if not case["requires_clarification"]:
            assert result["candidate_diseases"][0] in case["gold_diseases"], case["question_id"]
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"fictional_cases": len(results), "fixture_checks_passed": len(results),
                      "model_inference_performed": False, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
