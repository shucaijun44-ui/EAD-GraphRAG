"""Offline plumbing for fictional fixtures, separate from neural inference.

Candidate accumulation follows the archived relation/support formula. The
original PPR tail, document fusion, disease fusion and withholding functions
are then used directly. Mentions and cross-encoder outputs are fixture inputs.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from . import core


def candidates_from_triples(linked_entities, triples, crops):
    """In-memory adaptation of infer_disease_by_graph_symptoms for test data."""
    hosts = defaultdict(set)
    for triple in triples:
        if triple["relation"] == "belongs_to_crop":
            hosts[triple["head"]].add(triple["tail"])
    totals = defaultdict(float)
    support = defaultdict(set)
    allowed = {"symptom_of", "environment_causes", "caused_by_environment"}
    for entity in linked_entities:
        if core.is_real_disease(entity):
            continue
        for triple in triples:
            if triple["head"] != entity or triple["relation"] not in allowed:
                continue
            disease = triple["tail"]
            if not core.is_real_disease(disease):
                continue
            if crops and not hosts[disease].intersection(crops):
                continue
            totals[disease] += core.REL_WEIGHTS.get(triple["relation"], 1.0)
            support[disease].add(entity)
    weighted = {d: score * 2.0 ** (len(support[d]) - 1) for d, score in totals.items()}
    denominator = sum(weighted.values())
    return [
        {"disease": d, "score": score, "support": len(support[d]),
         "relative_score": round(score / denominator, 4) if denominator else 0.0}
        for d, score in sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    ][:core.SYSTEM_CONFIG["DIFFERENTIAL_TOP_K"]]


def fixture_paths(triples, linked, candidates, crops):
    """One-hop synthetic paths, not a replacement for the Neo4j traversal."""
    hosts = defaultdict(set)
    for triple in triples:
        if triple["relation"] == "belongs_to_crop":
            hosts[triple["head"]].add(triple["tail"])
    paths = []
    seen = set()
    seeds = set(linked) | set(candidates)
    for t in triples:
        h, rel, tail = t["head"], t["relation"], t["tail"]
        if rel in {"belongs_to_crop", "has_disease", "belongs_to_category"}:
            continue
        if not {h, tail}.intersection(seeds):
            continue
        if any(crops and hosts[n] and not hosts[n].intersection(crops) for n in (h, tail)):
            continue
        text = f"{h} → {rel} → {tail}"
        if text not in seen:
            seen.add(text)
            paths.append(([h, tail], [rel], text))
    return paths


class FixtureCrossEncoder:
    """Supplied synthetic scores; this class performs no model inference."""
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict(self, pairs):
        if len(pairs) != len(self.scores):
            raise ValueError("Fixture score count must match document count")
        return self.scores


def predict_fixture(observations, triples, chunks):
    # Labels, expected decisions, and case identifiers are not accepted here.
    allowed = {"question", "crop", "linked_entities", "entity_scores", "fixture_crossencoder_scores"}
    if set(observations) - allowed:
        raise ValueError("Only inference-side fixture inputs may be supplied")
    question = observations["question"]
    linked = list(dict.fromkeys(observations.get("linked_entities", [])))
    crops = observations.get("crop", [])
    initial = candidates_from_triples(linked, triples, crops)
    candidates = [item["disease"] for item in initial]
    local = fixture_paths(triples, linked, candidates, crops)
    paths, _, used = core.score_local_paths(
        local, list(dict.fromkeys(linked + candidates)), candidates,
        observations.get("entity_scores", {}),
        {item["disease"]: item["relative_score"] for item in initial}, "symptom",
    )
    documents = [{"doc": item["text"], "meta": {
        "disease": item["disease"], "crop": item["crop"], "type": item["type"]
    }} for item in chunks]
    reranker = FixtureCrossEncoder(observations["fixture_crossencoder_scores"])
    ranked_docs = core.decoupled_rerank_system(
        question, documents, paths, reranker, core.SYSTEM_CONFIG["FINAL_TOP_N"]
    )
    fused = core.rescore_differential_diagnosis(initial, ranked_docs, paths)
    withheld, margin = core.assess_diagnosis_uncertainty(fused)
    ranking = [item["disease"] for item in fused]
    if not ranking:
        # Suggestions do not change the empty-structured-candidate decision.
        ranking = list(dict.fromkeys(item["meta"]["disease"] for item in ranked_docs))[:3]
    return {
        "candidate_diseases": ranking,
        "initial_candidate_diseases": candidates,
        "differential_trace": fused,
        "uncertain": bool(withheld), "support_margin": margin,
        "kg_paths": paths, "graph_used": used,
        "execution_kind": "synthetic fixture replay; predefined mentions and CE outputs",
    }
