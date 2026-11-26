"""Optional resource-backed integration of the extracted original functions.

This new integration is not the frozen historical experiment runner. It uses
local model directories, caller-provided evidence and a read-only Neo4j session.
No model download, graph import, persistent-index reset or API billing occurs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def load_rows(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    rows = json.loads(text) if path.suffix.lower() == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of records: {path.name}")
    return rows


class _ReadOnlyDriver:
    def __init__(self, driver):
        self.driver = driver

    def session(self, **kwargs):
        kwargs.setdefault("database", "neo4j")
        kwargs["default_access_mode"] = "READ"
        return self.driver.session(**kwargs)


class ResourceRuntime:
    """Single-process diagnostic integration; resources are never distributed."""
    def __init__(self, resources_path, paper_config=None):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from neo4j import GraphDatabase
        from rank_bm25 import BM25Okapi
        import chromadb
        import jieba
        import torch

        path = Path(resources_path).resolve()
        resource = json.loads(path.read_text(encoding="utf-8"))
        root = Path(__file__).resolve().parents[1]

        def local(key):
            value = Path(resource[key])
            value = value if value.is_absolute() else root / value
            if not value.exists():
                raise FileNotFoundError(f"Required local resource is absent: {key}")
            return value.resolve()

        locations = {key: local(key) for key in ("kg", "chunks", "encoder", "reranker", "llm")}
        for variable in ("EAD_NEO4J_URI", "EAD_NEO4J_USER", "EAD_NEO4J_PASSWORD"):
            if not os.environ.get(variable):
                raise ValueError(f"Set {variable} in your local environment")
        spec = importlib.util.spec_from_file_location("ead_resource_core", Path(__file__).with_name("core.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.configure(paper_config)
        self.m = module
        self.driver = GraphDatabase.driver(
            os.environ["EAD_NEO4J_URI"],
            auth=(os.environ["EAD_NEO4J_USER"], os.environ["EAD_NEO4J_PASSWORD"]),
        )
        self.driver.verify_connectivity()
        module.graph_driver = _ReadOnlyDriver(self.driver)
        module.torch, module.jieba = torch, jieba
        module.device = torch.device(resource.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        module.embed_model = SentenceTransformer(str(locations["encoder"]), device=str(module.device), local_files_only=True)
        module.reranker_model = CrossEncoder(str(locations["reranker"]), max_length=512, device=str(module.device), local_files_only=True)
        triples = load_rows(locations["kg"])
        entities = {str(row[field]).strip() for row in triples for field in ("head", "tail", "subject", "object") if row.get(field)}
        # Same name-length filter as the archived dictionary loader, with explicit
        # tie ordering to make the integration deterministic across Python runs.
        module.ENTITY_DICT = sorted((e for e in entities if 2 <= len(e) <= 12), key=lambda e: (-len(e), e))
        if not module.ENTITY_DICT:
            raise ValueError("No usable graph entities were supplied")
        module.BM25_MODEL = BM25Okapi([list(jieba.cut(e)) for e in module.ENTITY_DICT])
        with torch.no_grad():
            module.ENTITY_EMBEDDINGS = torch.tensor(
                module.embed_model.encode(module.ENTITY_DICT, normalize_embeddings=True, show_progress_bar=False),
                dtype=torch.float32, device=module.device,
            )
        module.ENTITY_PROFILES, module.ENTITY_TYPES, module.ENTITY_CROPS = module.build_entity_profiles(module.ENTITY_DICT)
        self.tokenizer = AutoTokenizer.from_pretrained(str(locations["llm"]), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(locations["llm"]), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16 if module.device.type == "cuda" else torch.float32,
        ).to(module.device).eval()
        self.client = chromadb.EphemeralClient()
        module.collection = self.client.create_collection("ead_evidence", metadata={"hnsw:space": "cosine"})
        chunks = load_rows(locations["chunks"])
        ids = set()
        for item in chunks:
            text = item["text"]
            chunk_id = hashlib.md5(text.encode()).hexdigest()
            if chunk_id in ids:
                continue
            ids.add(chunk_id)
            module.collection.add(
                ids=[chunk_id], documents=[text],
                embeddings=[module.embed_model.encode(text, normalize_embeddings=True).tolist()],
                metadatas=[{key: str(item[key]) for key in ("type", "disease", "crop")}],
            )
        if not ids:
            raise ValueError("At least one evidence chunk is required")

    def predict(self, question):
        """Consume query text only; no gold label or expected decision is used."""
        m = self.m
        started = time.perf_counter()
        embedding = m.embed_model.encode(question, normalize_embeddings=True).tolist()
        parsed = m.llm_query_parser(question, self.model, self.tokenizer, m.device)
        keywords = m.augment_with_literal_entity_mentions(question, parsed["expanded_keywords"])
        scores, details = m.semantic_entity_linking(
            keywords, m.ENTITY_DICT, return_scores=True, return_details=True,
            query_text=question, typed_mentions=parsed.get("typed_mentions", []),
        )
        linked = list(scores)
        crops = m.resolve_taxonomy_crops(linked)
        differential = []
        for entity in linked:
            if any(word in entity for word in ["病斑", "斑点", "霉层", "症状", "点", "条斑", "轮纹", "萎蔫"]):
                continue
            if m.is_real_disease(entity):
                differential = [{"disease": entity, "score": 1.0, "support": 1, "relative_score": 1.0}]
                break
        if not differential:
            differential = m.infer_disease_by_graph_symptoms(linked, crops, m.SYSTEM_CONFIG["DIFFERENTIAL_TOP_K"])
        initial = [item["disease"] for item in differential]
        search = list(dict.fromkeys(linked + initial))
        seeds = m.select_graph_traversal_seeds(linked, details, initial, m.SYSTEM_CONFIG.get("GRAPH_QUERY_ENTITY_SEEDS", 4))
        paths, _, _ = m.graph_search_adaptive(
            search, parsed["dynamic_hop"], m.REL_WEIGHTS, question, crops,
            parsed["query_intent_type"], initial, scores,
            {item["disease"]: item["relative_score"] for item in differential}, seeds,
        )
        raw = m.vector_search_adaptive(
            question, embedding, target_crops=crops, query_type=parsed["query_intent_type"],
            top_k=m.SYSTEM_CONFIG["VECTOR_TOP_K"],
        )
        documents = m.decoupled_rerank_system(question, raw, paths, m.reranker_model, m.SYSTEM_CONFIG["FINAL_TOP_N"])
        differential = m.rescore_differential_diagnosis(differential, documents, paths)
        uncertain, margin = m.assess_diagnosis_uncertainty(differential)
        ranking = [item["disease"] for item in differential]
        if not ranking:
            ranking = list(dict.fromkeys(item["meta"].get("disease") for item in documents if item.get("meta", {}).get("disease")))[:3]
        return {
            "candidate_diseases": ranking, "initial_candidate_diseases": initial,
            "differential_trace": differential, "uncertain": bool(uncertain),
            "support_margin": margin, "linked_entities": linked, "kg_paths": paths,
            "retrieved_chunk_ids": [hashlib.md5(item["doc"].encode()).hexdigest() for item in documents],
            "clarification": m.build_kg_clarification_question(differential) if uncertain else None,
            "latency_seconds": time.perf_counter() - started,
            "execution_kind": "new resource integration; not an archived experiment replay",
        }

    def close(self):
        self.driver.close()
