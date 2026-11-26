# Data inventory and access boundaries

| Resource | Manuscript scope | Public deposit | Access classification |
| --- | --- | --- | --- |
| Domain source documents and evidence text | 322 disease entries; 579 attribute-level evidence units | Schema and newly authored fictional units only | Third-party source permissions not verified for redistribution; excluded |
| Complete domain graph | 5,641 entities; 14,618 directed triples | Schema and small fictional graph only | Full graph excluded; derivation alone does not establish redistribution permission |
| Controlled internal benchmark | 894 queries; 268 validation and 626 test | No study questions, labels, IDs or full splits | Derived-input permissions not verified for redistribution; excluded |
| External consultations | 181 single-disease consultations used for ranking evaluation | No consultation inputs, expert response text, HTML, source URLs or case labels | Third-party source material and associated records excluded |
| Original prediction logs | Per-query rankings, evidence and traces | No per-case outputs | Excluded to avoid exposing source content, source metadata and complete study records |
| Aggregate results | Nine selected experiment tables | Numeric aggregates, method/policy labels and existing aggregate intervals | Included with column allowlists and provenance |
| Headline ranking counts | Internal n=174; external n=181 | Two rank-histogram rows | Aggregate-only point-estimate verification |
| Fine-tuning triples and fine-tuned encoder | Agriculture-adapted BGE | Training recipe description only | Training data/checkpoint rights and original environment not verified; excluded |
| Upstream pretrained model weights | BGE and Qwen resources | No weights | Obtain from the upstream provider under its terms |
| Synthetic fixtures | Eight newly authored fictional cases | Complete tiny fictional fixture set | Demonstration inputs, not a sample from the research datasets |

The quoted resource sizes describe the manuscript; they do not describe the contents of this deposit. No complete research dataset or complete study knowledge graph is uploaded. Original source text is not made redistributable by masking names, rewriting a question or storing it in a private repository.

No controlled-access service, granted third-party redistribution permission or on-request delivery guarantee has been verified for the excluded resources. This repository does not promise that restricted originals can be provided by the authors. Any future access route must specify the relevant rights holder, permitted purpose and applicable conditions before materials are supplied.

Published point estimates can be inspected using `results/headline_rank_counts.csv`, `results/T09_PrimaryD1_283.csv` and `scripts/recompute_headlines.py`. Full historical reruns require additional resources described in `REPRODUCIBILITY.md`. The public histograms cannot recover original query text, disease labels, case identities or cluster assignments.
