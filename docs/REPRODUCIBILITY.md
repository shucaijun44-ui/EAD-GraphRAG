# Reproduction scope and execution

## Verified in the public package

The offline route runs the selected core functions against fictional fixture inputs. `scripts/recompute_headlines.py` calculates diagnostic point estimates from aggregate rank histograms derived from archived predictions, and acceptance accuracy/coverage from archived aggregate counts. Fingerprints identify the archival inputs used to produce those counts. None of the corresponding case-level input files is redistributed.

The public aggregate tables retain historical confidence intervals where present. Recomputing those intervals, paired comparisons or original significance tests requires the original paired outcomes and cluster assignments. Aggregate histograms do not preserve that information.

## Paper configuration

`configs/paper.json` explicitly sets the manuscript's final D1 policy: at least two distinct supporting entities and pre-fusion support margin 0.20. The graph propagation coefficient is 0.85, candidate restart contribution 1.00 and candidate-path multiplier 1.10. Document fusion weights are 0.60/0.40; disease fusion weights are 0.40/0.30/0.30.

The original archival runtime/configuration also contained earlier policy defaults. The release makes the final manuscript values explicit. D1 was applied to archived rankings as a decision overlay in the source results; the release does not relabel that overlay as a new end-to-end experiment.

The original document reranker has a scalar-type-dependent normalization branch. It is preserved for traceability: Python `float` values are passed through a sigmoid, while NumPy scalar outputs follow the original direct conversion branch. Changing model-library versions or score types can therefore change the scores. The fixture uses explicit NumPy `float32` scores. No claim of invariant numerical equivalence across dependency versions is made.

The archive also contained optional disease-level cross-encoder ranking and a separate post-hoc graph evidence display selector. These are omitted from this minimal package. The optional neural wrapper returns the original traversal paths; it does not recreate the historical display selector. Strict retrieval endpoint equivalence would need the original runner and evidence-selection configuration.

## Optional local resource integration

This route is supplied as a new integration of the extracted functions. It has been syntax checked, but the original fine-tuned model, graph service and complete authorized study resources were not loaded during package preparation.

1. Prepare the resources described in `DATA_SCHEMA.md` under a locally excluded directory such as `restricted/`. Use an isolated Neo4j database named `neo4j`, with the documented labels, relationship directions and `name` properties. Importing a graph and obtaining permissions are separate tasks; this package does not import or reset a database.
2. Obtain the original fine-tuned `bge-small-zh-agri` encoder, BGE reranker and Qwen2.5-7B-Instruct model using legitimate access routes. Model weights and training records are not part of this deposit. Substituting the unmodified BGE model is a changed experiment.
3. Install the optional dependencies in a separate environment:

   ```bash
   python -m pip install -r requirements-neural.txt
   ```

4. Make a local copy of `configs/resources.example.json`, point it to your authorized graph/chunks/model directories and choose `cpu` or `cuda`. Paths are resolved relative to the repository root. Supply `EAD_NEO4J_URI`, `EAD_NEO4J_USER` and `EAD_NEO4J_PASSWORD` as local environment variables. Do not put values into tracked files.
5. Prepare query-only JSONL records and run:

   ```bash
   python scripts/run_inference.py --resources restricted/resources.json --queries restricted/queries.jsonl --output outputs/predictions.jsonl
   python scripts/evaluate.py --gold restricted/gold.jsonl --predictions outputs/predictions.jsonl --output outputs/evaluation.json
   ```

The inference input accepts only `question_id` and `question`. Labels remain separate. Missing or failed predictions remain in the evaluation denominator. The integration builds an ephemeral vector collection without deleting or replacing any persistent index. All model loading is local; no automatic model download is used.

The wrapper deterministically orders equal-length entity names, whereas the original loader used set-derived tie order. It also supplies a fresh ephemeral index and omits the response-presentation UI. These changes are disclosed integration choices, not evidence that historical results have been reproduced exactly. Qwen is used for query parsing; the free-form answer-presentation layer is omitted.

## Encoder training recipe reported in the manuscript

The manuscript reports query–positive–hard-negative supervision with Multiple Negatives Ranking Loss in Sentence-Transformers, batch size 32, learning rate 2e-5, four epochs and 10% warmup; the best encoder was selected by validation Recall@10. This is a record of the reported recipe. A verified original training script, exact training examples, best checkpoint and full environment lock are not supplied here, so this description does not establish exact encoder retraining reproducibility.

## Baselines and statistics

The nine aggregate tables preserve the comparison configurations and results available in the archive, including weaker methods and failed-query denominators. Full LightRAG/HippoRAG installations and historical baseline runners are outside this minimal deposit. The manuscript reports a shared Qwen2.5-7B reader at temperature 0 and maximum output 1,024 tokens, with original and matched evidence-budget comparisons kept distinct.

The archived generic evaluator supplies exact directed-triple matching, cluster-bootstrap summaries, cluster sign-flip tests and Holm adjustment functions. Its legacy substring metric is named separately and must not be substituted for the strict endpoint. The command-line evaluator provides point estimates for coverage and accepted accuracy; it does not reconstruct the original D1 cluster intervals or the historical Holm test families from aggregate tables.
