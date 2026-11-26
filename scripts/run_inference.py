"""Optional inference on the user's locally authorized resources."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ead_graphrag.runtime import ResourceRuntime, load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-config", type=Path, default=ROOT / "configs/paper.json")
    args = parser.parse_args()
    queries = load_rows(args.queries)
    ids = set()
    for row in queries:
        if set(row) != {"question_id", "question"}:
            raise ValueError("Inference input accepts question_id and question only; keep labels in a separate file")
        if not isinstance(row["question_id"], str) or not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Each query needs a string identifier and nonempty question")
        if row["question_id"] in ids:
            raise ValueError("Duplicate query identifier")
        ids.add(row["question_id"])
    if args.output.exists():
        raise FileExistsError("Choose a new output filename to preserve previous results")
    with redirect_stdout(sys.stderr):
        runtime = ResourceRuntime(args.resources, args.paper_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            for row in queries:
                try:
                    with redirect_stdout(sys.stderr):
                        prediction = runtime.predict(row["question"])
                except Exception as error:
                    # Error class is enough for scoring; do not dump a connection
                    # string or private query into an exception message.
                    prediction = {"error": type(error).__name__, "candidate_diseases": []}
                prediction["question_id"] = row["question_id"]
                handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
    finally:
        runtime.close()
    print(json.dumps({"queries": len(queries), "output": str(args.output)}))


if __name__ == "__main__":
    main()
