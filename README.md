# EAD-GraphRAG

Minimal research package for **EAD-GraphRAG: A candidate-guided knowledge graph–text evidence integration framework for selective crop disease diagnosis**.

This package contains selected original algorithm functions, a manuscript-aligned D1 configuration, the archived evaluation functions, nine aggregate result tables, two rank-histogram rows and eight fictional demonstration cases. It excludes the complete study datasets, original consultations, third-party text evidence, model weights, private indexes, server credentials, web application and development history.

## Run the offline demonstration

Use Python 3.10 or later in a separate environment:

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
python scripts/recompute_headlines.py
python -m unittest discover -s tests -v
python scripts/check_release.py
```

The demonstration uses **predefined synthetic entity mentions and cross-encoder scores**. It executes candidate weighting, the original local-graph PPR scoring block, document fusion, disease fusion and the D1 acceptance rule. It does not run the neural entity linker, Qwen model or original benchmark, and its eight checks are software checks rather than scientific performance estimates.

The headline script independently reconstructs the ranking point estimates from two aggregate count rows: 92.53%/95.98%/94.16% internally and 67.40%/80.11%/73.30% externally for Acc@1/Recall@3/MRR@3. It also checks the mixed-task D1 acceptance counts: 112 correct diagnoses among 116 accepted cases, from 283 cases overall. These are computations from archived outcomes, not new model results.

## What can be reproduced

| Route | Included inputs | Scope |
| --- | --- | --- |
| Offline demonstration | Newly authored fictional graph, text and cases | Core numerical mechanisms and D1 behavior |
| Headline verification | Aggregate rank counts and decision counts | Published point estimates, without original case content |
| Local inference/evaluation | Code and schemas; resources must be provided separately | New runs on locally authorized resources |
| Exact historical experiment | Complete benchmark, corpus, graph snapshot, fine-tuned encoder, indexes and original server environment are not included | Cannot be completed from this public package alone |

The selected original function bodies are preserved and fingerprinted in [SOURCE_PROVENANCE.json](docs/SOURCE_PROVENANCE.json). The new resource integration and offline adapters are labeled separately. This package is not represented as an independently verified, frozen copy of the original experiment environment.

## Main files

| Path | Content |
| --- | --- |
| `ead_graphrag/core.py` | Selected original linking, candidate, graph retrieval, fusion and withholding functions |
| `ead_graphrag/runtime.py` | Optional integration using local models and a read-only Neo4j connection |
| `ead_graphrag/evaluation.py` | Archived scoring and cluster-bootstrap functions |
| `configs/paper.json` | D1 support ≥ 2; margin ≥ 0.20; fusion weights 0.40/0.30/0.30 |
| `data/synthetic/` | Eight fictional cases, three text units and a small fictional graph |
| `results/` | Aggregate tables and rank histograms; no per-case study data |
| `docs/` | Reproduction scope, resource schemas, provenance and data-access boundaries |

## Use local resources

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [DATA_SCHEMA.md](docs/DATA_SCHEMA.md). The optional integration requires local BGE encoder/reranker and Qwen model directories, authorized evidence chunks and graph data, and a compatible Neo4j database. Installing `requirements-neural.txt` alone does not supply these resources. The optional neural route has not been executed as part of this public-package verification.

## Data and rights

See [DATA_ACCESS.md](docs/DATA_ACCESS.md). The fictional data are newly authored test fixtures; they are not sampled, masked or paraphrased consultation records. Aggregate exports retain all methods/rows in the nine selected experiment tables. Individual errors, cases, questions, source URLs and complete resources are omitted.

No software or data reuse license has been selected for this deposit; see [RIGHTS.md](RIGHTS.md). Public access to a repository does not establish redistribution rights over excluded third-party materials. Cite the software metadata in [CITATION.cff](CITATION.cff).
