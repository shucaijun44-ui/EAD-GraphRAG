# Local resource formats

All examples below refer to the fictional software fixtures. They do not grant access to the study resources. Use UTF-8 JSON/JSONL. Keep locally authorized research inputs outside the release manifest.

## Directed graph triples

Each record contains `head`, `relation`, `tail`:

```json
{"head":"叶背示例点","relation":"symptom_of","tail":"番茄示例甲病"}
```

Neo4j nodes need a `name` property. Crop nodes use label `Crop`; disease/symptom/environment labels are used by the entity-profile construction. Relations are directed, including `symptom_of`, `has_symptom`, `environment_causes`, `caused_by_environment`, `belongs_to_crop` and the other configured relation types. Do not silently invert graph edges. The supplied integration reads an existing isolated database and does not create or modify these records.

## Attribute-level evidence units

```json
{"id":"synthetic-chunk-1","text":"虚构测试条目内容。","type":"symptom","disease":"番茄示例甲病","crop":"番茄"}
```

`type` is `symptom`, `cause` or `prevention`. The historical retrieval/evaluation code uses the MD5 of the exact UTF-8 text as the retrieval identifier, so changing spaces or text changes chunk alignment. The integration computes that identifier independently of an optional descriptive `id` field.

## Query-only inference input

```json
{"question_id":"local-example-1","question":"番茄出现环形示例斑及叶背示例点。"}
```

The inference CLI rejects extra fields, including gold labels. The fictional offline fixture additionally contains predefined mentions and CE scores, explicitly labeled as fixture inputs.

## Separate gold evaluation input

Required fields: `question_id`, `question`. Common optional fields are `case_family_id`, `case_id`, `subset`, `crop`, `gold_diseases`, `requires_clarification`, `gold_chunk_ids` and `gold_kg_triples`.

The scorer recognizes `ambiguous` and `ood_unanswerable` as withholding-positive subsets, and also honors `requires_clarification`. Supply original family/group identifiers when cluster statistics are needed; omit invented independence claims. Duplicate question text and family links are connected into the archived scorer's cluster definition.

## Prediction input

Predictions contain `question_id`, ranked `candidate_diseases` and a Boolean `uncertain`. Retrieval evaluation additionally accepts `retrieved_chunk_ids` and ordered `kg_paths`. Initial-candidate evaluation accepts `initial_candidate_diseases`.

```json
{"question_id":"local-example-1","candidate_diseases":["番茄示例甲病"],"initial_candidate_diseases":["番茄示例甲病"],"uncertain":false}
```

Represent a failed run with an `error` field. Do not discard failed rows before scoring. Missing predictions are treated as failures. Directed triples are matched exactly after expanding the first ten paths, not by searching for disconnected substrings.
