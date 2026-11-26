"""Selected original EAD-GraphRAG functions, without server initialization.

Function bodies are preserved from the archived runtime. See
docs/SOURCE_PROVENANCE.json for fingerprints and extraction boundaries.
Call configure() before use; runtime.py supplies optional model/database objects.
"""
from __future__ import annotations
from collections import defaultdict
from functools import wraps
from pathlib import Path
import copy
import json
import math
import re
import networkx as nx

DEBUG_LOG_OUTPUT = False


class _LazyTorch:
    def __getattr__(self, name):
        import torch as module
        return getattr(module, name)

    def inference_mode(self):
        # Preserve the original decorator while keeping offline imports light.
        def decorate(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                import torch as module
                with module.inference_mode():
                    return function(*args, **kwargs)
            return wrapped
        return decorate


torch = _LazyTorch()


def configure(config_path=None):
    global SYSTEM_CONFIG, VALIDATION_CONFIG, REL_WEIGHTS, INTENT_RELATION_BOOST
    global DEFAULT_DIAGNOSIS_FUSION_WEIGHTS
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "configs/paper.json"
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    VALIDATION_CONFIG = copy.deepcopy(config)
    SYSTEM_CONFIG = copy.deepcopy(config["system"])
    REL_WEIGHTS = copy.deepcopy(config["relation_weights"])
    INTENT_RELATION_BOOST = copy.deepcopy(config["intent_relation_boost"])
    DEFAULT_DIAGNOSIS_FUSION_WEIGHTS = copy.deepcopy(config["diagnosis_fusion_weights"])
    return config


configure()


def _stable_sigmoid(value):
    """Normalize arbitrary CrossEncoder logits to [0, 1] deterministically."""
    x = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-x))


def build_entity_profiles(entity_dict, batch_size=256, neighbor_limit=8):
    """Build compact KG descriptions used by the entity-linking cross-encoder.

    ReFinED-style type/description evidence is represented with Neo4j labels and
    a small one-hop neighborhood.  Missing KG metadata degrades gracefully to
    name-only profiles.
    """
    profiles = {name: f"实体：{name}" for name in entity_dict}
    types = {name: set() for name in entity_dict}
    crops = {name: set() for name in entity_dict}
    try:
        with graph_driver.session(database="neo4j") as session:
            for start in range(0, len(entity_dict), batch_size):
                names = entity_dict[start:start + batch_size]
                rows = session.run("""
                    UNWIND $names AS entity_name
                    MATCH (n {name: entity_name})
                    OPTIONAL MATCH (n)-[r]-(m)
                    WITH n, labels(n) AS node_labels,
                         collect(DISTINCT CASE
                             WHEN m.name IS NULL THEN NULL
                             ELSE type(r) + ':' + m.name
                         END) AS ctx
                    RETURN n.name AS name, node_labels, ctx
                """, names=names)
                for row in rows:
                    name = row["name"]
                    labels = [str(x) for x in (row["node_labels"] or [])]
                    ctx = [str(x) for x in (row["ctx"] or []) if x][:neighbor_limit]
                    types[name] = {x.lower() for x in labels}
                    profile = f"实体：{name}。类型：{'/'.join(labels) if labels else 'Unknown'}"
                    if ctx:
                        profile += "。图谱邻域：" + "；".join(ctx)
                    profiles[name] = profile
                    crops[name] = {c for c in SYSTEM_CONFIG["CROP_LIST"] if c in profile or c in name}
    except Exception as e:
        print("⚠️ 实体描述缓存构建失败，退化为名称级实体链接:", e)
    return profiles, types, crops


def build_vector_where(target_disease=None, target_crops=None, query_type=None):
    """Build a Chroma-compatible filter without nested invalid crop expressions."""
    conditions = []
    if target_disease:
        conditions.append({"disease": target_disease})
    if query_type:
        conditions.append({"type": query_type})
    if target_crops:
        crops = list(dict.fromkeys(target_crops))
        if len(crops) == 1:
            conditions.append({"crop": crops[0]})
        else:
            conditions.append({"crop": {"$in": crops}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def vector_search_adaptive(query, q_emb, target_disease=None, target_crops=None, query_type=None, top_k=40):
    conditions = []
    if target_disease:
        conditions.append({"disease": target_disease})
    if query_type:
        conditions.append({"type": query_type})
        
    if target_crops and len(target_crops) > 0:
        if len(target_crops) == 1:
            conditions.append({"crop": target_crops[0]})
        else:
            conditions.append({"crop": {"$and": [{"$or": [{"crop": c} for c in target_crops]}]}})

    where_condition = None
    if len(conditions) > 1:
        if "crop" in conditions[-1] and isinstance(conditions[-1]["crop"], dict) and "$and" in conditions[-1]["crop"]:
            crop_cond = conditions.pop()
            where_condition = {"$and": conditions + crop_cond["crop"]["$and"]}
        else:
            where_condition = {"$and": conditions}
    elif len(conditions) == 1:
        if "crop" in conditions[0] and isinstance(conditions[0]["crop"], dict) and "$and" in conditions[0]["crop"]:
            where_condition = conditions[0]["crop"]
        else:
            where_condition = conditions[0]

    res = collection.query(
        query_embeddings=[q_emb], 
        n_results=top_k, 
        include=["documents", "metadatas", "distances"],
        where=where_condition
    )
    
    if not res["documents"] or len(res["documents"][0]) == 0:
        if query_type and (target_disease or target_crops):
            print("⚠️ 属性路由双重锁定未召回，执行【第一阶降阶】：卸载属性限制，保持大类作物或疾病锁...")
            return vector_search_adaptive(query, q_emb, target_disease=target_disease, target_crops=target_crops, query_type=None, top_k=top_k)
        elif (target_disease or target_crops) and not query_type:
            print("⚠️ 作物群/疾病锁定仍无召回，执行【第二阶降阶】：完全解开刚性硬过滤，退化为全局语义语义检索...")
            return vector_search_adaptive(query, q_emb, target_disease=None, target_crops=None, query_type=None, top_k=top_k)
        return []
        
    out = []
    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"doc": d, "meta": m or {}, "dist": float(dist)})
    return out


def resolve_taxonomy_crops(linked_entities):
    resolved_crops = set()
    with graph_driver.session() as session:
        for ent in linked_entities:
            is_cat = session.run("MATCH (n:Category {name: $name}) RETURN n", name=ent)
            if list(is_cat):
                crops = session.run("""
                    MATCH (c:Crop)-[:belongs_to_category*1..2]->(cat:Category {name: $name})
                    RETURN DISTINCT c.name AS crop_name
                """, name=ent)
                for r in crops:
                    resolved_crops.add(r["crop_name"])
            
            is_crop = session.run("MATCH (n:Crop {name: $name}) RETURN n", name=ent)
            if list(is_crop):
                resolved_crops.add(ent)
                
    return list(resolved_crops)


def is_real_disease(name):
    # 💥 核心修复：建立硬红线黑名单，过滤用户意图疑问句中的泛指和噪音词，防止错锁
    invalid_disease_keywords = ["遭受病害", "什么病", "哪个病", "那种病", "发生病害", "诱发病害", "未知病害", "哪些病"]
    if any(kw in name for kw in invalid_disease_keywords):
        return False

    symptom_suffixes = ["斑", "点", "纹", "层", "表现", "部位", "叶", "果", "茎", "根", "花", "变化", "特征"]
    if (name.endswith("病") or name.endswith("瘟") or name.endswith("害") or "病毒" in name):
        if any(name.endswith(s) for s in symptom_suffixes):
            return False
        return True
    return False


def infer_disease_by_graph_symptoms(linked_entities, resolved_crops=None, top_k=None):
    """Return a differential diagnosis list instead of prematurely committing to one disease."""
    if not linked_entities:
        return []
        
    inferred_diseases = defaultdict(float)
    disease_co_occurrence = defaultdict(set)
    
    with graph_driver.session() as session:
        for ent in linked_entities:
            if is_real_disease(ent):
                continue
                
            if resolved_crops:
                query = """
                    MATCH (n)-[r:symptom_of|environment_causes|caused_by_environment]->(d)
                    WHERE n.name = $name
                    AND EXISTS {
                        MATCH (d)-[:belongs_to_crop]->(c:Crop)
                        WHERE c.name IN $allowed_crops
                    }
                    RETURN d.name AS disease_name, type(r) AS rel_type
                """
                result = session.run(query, name=ent, allowed_crops=resolved_crops)
            else:
                query = """
                    MATCH (n)-[r:symptom_of|environment_causes|caused_by_environment]->(d)
                    WHERE n.name = $name
                    RETURN d.name AS disease_name, type(r) AS rel_type
                """
                result = session.run(query, name=ent)
            
            for r in result:
                d_name = r["disease_name"]
                if is_real_disease(d_name):
                    weight = float(REL_WEIGHTS.get(r["rel_type"], 1.0))
                    inferred_diseases[d_name] += weight
                    disease_co_occurrence[d_name].add(ent)
                    
    if inferred_diseases:
        final_scores = {}
        for d_name, base_w in inferred_diseases.items():
            hit_seeds_count = len(disease_co_occurrence[d_name])
            final_scores[d_name] = base_w * (2.0 ** (hit_seeds_count - 1))
            
        sorted_diseases = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(final_scores.values())
        return [{"disease": name, "score": score, "support": len(disease_co_occurrence[name]),
                 "relative_score": round(score / total, 4) if total else 0.0}
                for name, score in sorted_diseases[:(top_k or SYSTEM_CONFIG["DIFFERENTIAL_TOP_K"])]]
    return []


def rescore_differential_diagnosis(differential_diagnosis, reranked_docs, kg_paths):
    """Fuse KG diagnosis prior, graph-path support and document evidence.

    Only existing differential candidates can be re-ranked.  This makes the
    diagnosis decision evidence-aware without allowing text retrieval to inject
    a new unsupported disease name.
    """
    if not differential_diagnosis or len(differential_diagnosis) <= 1:
        return differential_diagnosis
    candidates = [x["disease"] for x in differential_diagnosis]
    kg = {x["disease"]: float(x.get("relative_score", 0.0)) for x in differential_diagnosis}
    graph = {d: 0.0 for d in candidates}
    text_support = {d: 0.0 for d in candidates}

    for rank, path in enumerate(kg_paths or [], 1):
        discount = 1.0 / math.log2(rank + 1.0)
        for d in candidates:
            if d in path:
                graph[d] += discount

    for rank, item in enumerate(reranked_docs or [], 1):
        d = (item.get("meta") or {}).get("disease")
        if d in text_support:
            discount = 1.0 / math.log2(rank + 1.0)
            text_support[d] += float(item.get("final_composite_score", 0.0)) * discount

    def max_norm(scores):
        m = max(scores.values()) if scores else 0.0
        return {k: (v / m if m > 0 else 0.0) for k, v in scores.items()}
    kg_n, graph_n, text_n = max_norm(kg), max_norm(graph), max_norm(text_support)
    weights = VALIDATION_CONFIG.get("diagnosis_fusion_weights", DEFAULT_DIAGNOSIS_FUSION_WEIGHTS)

    rescored = []
    for item in differential_diagnosis:
        d = item["disease"]
        # validation_config historically used the key "ppr", while this V5
        # implementation called the same signal "graph". Accept both names.
        graph_weight = float(weights.get("graph", weights.get("ppr", 0.20)))
        fused = (float(weights.get("kg", 0.55)) * kg_n.get(d, 0.0) +
                 graph_weight * graph_n.get(d, 0.0) +
                 float(weights.get("text", 0.25)) * text_n.get(d, 0.0))
        new_item = dict(item)

        # Preserve the pre-fusion differential distribution for the uncertainty
        # head. Evidence fusion is a ranking score, not a calibrated confidence
        # probability, so it should not silently redefine query ambiguity.
        new_item["pre_fusion_relative_score"] = float(item.get("relative_score", 0.0))
        new_item["pre_fusion_support"] = int(item.get("support", 0))

        new_item["fused_score"] = fused
        new_item["evidence_breakdown"] = {"kg": kg_n.get(d, 0.0), "graph": graph_n.get(d, 0.0),
                                          "text": text_n.get(d, 0.0)}
        rescored.append(new_item)
    rescored.sort(key=lambda x: x["fused_score"], reverse=True)
    total = sum(x["fused_score"] for x in rescored)
    for x in rescored:
        x["relative_score"] = round(x["fused_score"] / total, 4) if total > 0 else x.get("relative_score", 0.0)
    return rescored


def assess_diagnosis_uncertainty(differential_diagnosis):
    """Estimate query-level ambiguity independently from the final ranking score.

    Final disease ranking may be changed by KG/PPR/Text evidence fusion. Those
    fused scores are useful for ranking but are not calibrated probabilities.
    When available, uncertainty therefore uses the pre-fusion structured
    differential distribution saved by ``rescore_differential_diagnosis``.
    This makes ranking and confidence two separate heads:

        diagnosis head   -> fused KG/PPR/Text ranking
        uncertainty head -> structured differential support + pre-fusion margin

    The decision rule itself is unchanged: insufficient graph support or a
    small Top-1/Top-2 margin triggers clarification.
    """
    if not differential_diagnosis:
        return True, None

    use_decoupled = bool(SYSTEM_CONFIG.get("USE_DECOUPLED_UNCERTAINTY", True))
    has_pre_fusion = all(
        "pre_fusion_relative_score" in x for x in differential_diagnosis
    )

    if use_decoupled and has_pre_fusion:
        # Reconstruct the structured differential view that existed before
        # diagnosis evidence fusion reordered the candidates.
        confidence_view = sorted(
            differential_diagnosis,
            key=lambda x: float(x.get("pre_fusion_relative_score", 0.0)),
            reverse=True,
        )
        score_key = "pre_fusion_relative_score"
        support_key = "pre_fusion_support"
    else:
        # Backward-compatible path for non-fused variants.
        confidence_view = differential_diagnosis
        score_key = "relative_score"
        support_key = "support"

    top_support = int(confidence_view[0].get(support_key, confidence_view[0].get("support", 0)))
    if top_support < SYSTEM_CONFIG["MIN_DIAGNOSIS_SUPPORT"]:
        return True, None

    if len(confidence_view) < 2:
        return False, None

    margin = (
        float(confidence_view[0].get(score_key, 0.0))
        - float(confidence_view[1].get(score_key, 0.0))
    )
    return margin < SYSTEM_CONFIG["DIAGNOSIS_MARGIN"], margin


def build_kg_clarification_question(differential_diagnosis):
    """Ask about symptoms that discriminate the two leading KG candidates."""
    if not differential_diagnosis:
        return "当前症状未在图谱中形成足够的病害证据，请补充作物、发病部位、病斑形状及发生环境。"
    if len(differential_diagnosis) < 2:
        disease = differential_diagnosis[0]["disease"]
        return (f"当前仅有【{disease}】这一条低支持度候选，尚不足以确诊。"
                "请补充病斑部位、形状、颜色及叶背是否有霉层等信息。")
    first, second = differential_diagnosis[0]["disease"], differential_diagnosis[1]["disease"]
    query = """
        MATCH (d {name: $disease})-[:has_symptom]->(s) RETURN DISTINCT s.name AS symptom
        UNION
        MATCH (s)-[:symptom_of]->(d {name: $disease}) RETURN DISTINCT s.name AS symptom
    """
    try:
        with graph_driver.session() as session:
            symptoms_1 = {row["symptom"] for row in session.run(query, disease=first) if row["symptom"]}
            symptoms_2 = {row["symptom"] for row in session.run(query, disease=second) if row["symptom"]}
    except Exception as e:
        print("⚠️ KG 澄清属性查询失败：", e)
        return None
    unique_1, unique_2 = sorted(symptoms_1 - symptoms_2), sorted(symptoms_2 - symptoms_1)
    if unique_1 and unique_2:
        return (f"目前无法在【{first}】和【{second}】之间可靠区分。"
                f"请观察：是更符合“{unique_1[0]}”，还是“{unique_2[0]}”？")
    if unique_1:
        return f"为区分【{first}】与【{second}】，请确认是否出现“{unique_1[0]}”。"
    if unique_2:
        return f"为区分【{first}】与【{second}】，请确认是否出现“{unique_2[0]}”。"
    return f"【{first}】与【{second}】的现有图谱症状重叠较多；请补充病斑部位、形状及叶背是否有霉层。"


def _heuristic_primary_intent(query):
    """Deterministic fallback only; the LLM remains the primary intent parser."""
    if any(w in query for w in ["防治", "用药", "药剂", "怎么治", "喷什么", "喷施", "治疗"]):
        return "prevention"
    if any(w in query for w in ["原因", "病因", "为什么", "诱因", "怎么引起", "传播"]):
        return "cause"
    return "symptom"


@torch.inference_mode()
def llm_query_parser(query, model, tokenizer, device):
    prompt_template = f"""<|im_start|>system
你是农业病害知识图谱查询解析器。只输出合法JSON，不要Markdown，不要解释。
<|im_end|>
<|im_start|>user
请学习以下三个均衡示例：

示例1："番茄叶片出现同心轮纹和褐色病斑，是什么病？"
{{"expanded_keywords":["番茄","叶片","同心轮纹","褐色病斑"],"typed_mentions":[{{"text":"番茄","type":"crop"}},{{"text":"叶片","type":"symptom"}},{{"text":"同心轮纹","type":"symptom"}},{{"text":"褐色病斑","type":"symptom"}}],"query_intent_type":"symptom","dynamic_hop":2}}

示例2："黄瓜霜霉病为什么在高湿条件下更严重？"
{{"expanded_keywords":["黄瓜","霜霉病","高湿"],"typed_mentions":[{{"text":"黄瓜","type":"crop"}},{{"text":"霜霉病","type":"disease"}},{{"text":"高湿","type":"environment"}}],"query_intent_type":"cause","dynamic_hop":2}}

示例3："辣椒叶片出现病斑该怎么防治，用什么药？"
{{"expanded_keywords":["辣椒","叶片","病斑"],"typed_mentions":[{{"text":"辣椒","type":"crop"}},{{"text":"叶片","type":"symptom"}},{{"text":"病斑","type":"symptom"}}],"query_intent_type":"prevention","dynamic_hop":2}}

当前真实问题："{query}"
要求：
1. expanded_keywords 和 typed_mentions 只能抽取用户原文真实出现的作物、疾病（仅当用户明确提及）、症状、环境或药剂词；禁止猜测疾病。
2. type 只能是 crop/disease/symptom/environment/chemical/unknown。
3. query_intent_type 只能是 symptom/cause/prevention；“是什么病/诊断”归入 symptom。
4. dynamic_hop 只能是1、2或3；简单单跳问题优先1-2跳，复杂因果/防治链可2-3跳。

只输出JSON：
<|im_end|>
<|im_start|>assistant
"""
    try:
        inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        raw_config = json.loads(json_match.group(0) if json_match else response)

        expanded = [str(x).strip() for x in raw_config.get("expanded_keywords", []) if str(x).strip()]
        if not expanded:
            expanded = [query]
        intent = raw_config.get("query_intent_type")
        if intent not in {"symptom", "cause", "prevention"}:
            intent = _heuristic_primary_intent(query)
        try:
            hop = int(raw_config.get("dynamic_hop", 2))
        except Exception:
            hop = 2
        hop = max(1, min(3, hop))

        typed = []
        valid_types = {"crop", "disease", "symptom", "environment", "chemical", "unknown"}
        for item in raw_config.get("typed_mentions", []):
            if not isinstance(item, dict):
                continue
            mention = str(item.get("text", "")).strip()
            mtype = str(item.get("type", "unknown")).strip().lower()
            if mention and mention in query:
                typed.append({"text": mention, "type": mtype if mtype in valid_types else "unknown"})
        if not typed:
            typed = [{"text": kw, "type": "crop" if kw in SYSTEM_CONFIG["CROP_LIST"] else "unknown"}
                     for kw in expanded]

        parsed_config = {
            "expanded_keywords": expanded,
            "typed_mentions": typed,
            "query_intent_type": intent,
            "dynamic_hop": hop,
        }
        print(f"🧠 LLM解析成功: intent={intent}, mentions={[(x['text'], x['type']) for x in typed]}")
        return parsed_config
    except Exception as e:
        print("⚠️ 意图解析失败，使用确定性规则兜底:", e)
        return {
            "expanded_keywords": [query],
            "typed_mentions": [{"text": query, "type": "unknown"}],
            "query_intent_type": _heuristic_primary_intent(query),
            "dynamic_hop": 2,
        }


def _mention_type_prior(mention_type, entity_name):
    labels = ENTITY_TYPES.get(entity_name, set())
    if not labels or mention_type in (None, "unknown"):
        return 0.85
    aliases = {
        "crop": {"crop"},
        "disease": {"disease", "pest"},
        "symptom": {"symptom", "feature"},
        "environment": {"environment", "condition"},
        "chemical": {"chemical", "medicine", "pesticide"},
    }
    wanted = aliases.get(mention_type, set())
    if any(label in wanted for label in labels):
        return 1.0
    return 0.40


def _crop_compatibility_prior(active_crops, entity_name):
    if not active_crops:
        return 0.90
    entity_crops = ENTITY_CROPS.get(entity_name, set())
    if not entity_crops:
        return 0.85
    return 1.0 if set(active_crops) & entity_crops else 0.25


def augment_with_literal_entity_mentions(query_text, expanded_keywords, max_extra=5):
    """Add canonical KG names literally present in the user query.

    This is a high-precision lexical recall path and does not generate new text.
    Longest mentions are preferred to avoid flooding the linker with nested terms.
    """
    expanded = list(dict.fromkeys(expanded_keywords))
    if not query_text:
        return expanded
    present = [e for e in ENTITY_DICT if len(e) >= 2 and e in query_text and e not in expanded]
    present.sort(key=len, reverse=True)
    chosen = []
    for ent in present:
        if any(ent in longer for longer in chosen):
            continue
        chosen.append(ent)
        if len(chosen) >= max_extra:
            break
    return list(dict.fromkeys(expanded + chosen))


def semantic_entity_linking(expanded_keywords, entity_dict, threshold=None, return_scores=False,
                            query_text="", typed_mentions=None, return_details=False):
    """Batched two-stage, type-aware entity linker.

    Stage 1 retrieves a small candidate set for every mention with
    dense + BM25 + lexical matching.
    Stage 2 reranks *all mention-candidate pairs in one CrossEncoder batch*.
    This preserves the V5-E scoring rule while avoiding one predict() call
    per mention.

    return_details=True additionally returns per-entity metadata used only
    for graph-seed selection.  The default return contract is unchanged.
    """
    if threshold is None:
        threshold = float(SYSTEM_CONFIG.get("SIM_THRESHOLD", 0.70))

    typed_mentions = typed_mentions or []
    type_by_text = {
        str(x.get("text", "")): str(x.get("type", "unknown"))
        for x in typed_mentions if isinstance(x, dict)
    }
    active_crops = [
        c for c in SYSTEM_CONFIG["CROP_LIST"]
        if c in query_text or any(c in kw for kw in expanded_keywords)
    ]

    valid_kws = list(dict.fromkeys(
        kw.strip() for kw in expanded_keywords if len(kw.strip()) >= 2
    ))
    if not valid_kws:
        empty = {} if return_scores else []
        return (empty, {}) if return_details else empty

    kw_embs_all = torch.tensor(
        embed_model.encode(
            valid_kws,
            normalize_embeddings=True,
            show_progress_bar=False
        ),
        dtype=torch.float32,
        device=device
    )
    all_dense_scores = torch.matmul(ENTITY_EMBEDDINGS, kw_embs_all.t())
    fw = VALIDATION_CONFIG["fusion_weights"]

    # -------- Stage 1: candidate generation for every mention --------
    mention_groups = []
    all_pairs = []
    pair_slots = []

    for kw_idx, kw in enumerate(valid_kws):
        mention_type = type_by_text.get(
            kw,
            "crop" if kw in SYSTEM_CONFIG["CROP_LIST"] else "unknown"
        )
        is_pure_crop = kw in SYSTEM_CONFIG["CROP_LIST"]
        dense_scores = all_dense_scores[:, kw_idx]

        query_tokens = list(jieba.cut(kw))
        raw_bm_scores = BM25_MODEL.get_scores(query_tokens)
        bm_scores = torch.tensor(
            raw_bm_scores,
            dtype=torch.float32,
            device=device
        )
        max_bm, min_bm = bm_scores.max(), bm_scores.min()
        if max_bm > min_bm:
            bm_scores = (
                bm_scores - min_bm
            ) / (max_bm - min_bm + 1e-8)
        else:
            bm_scores = torch.zeros_like(bm_scores)

        candidates = []
        for idx, ent in enumerate(entity_dict):
            # Only hard-reject an explicit crop contradiction.
            if active_crops and any(
                crop in ent and crop not in active_crops
                for crop in SYSTEM_CONFIG["CROP_LIST"]
            ):
                continue

            dense = float(dense_scores[idx])
            bm = float(bm_scores[idx])

            exact = 0.0
            if kw == ent:
                exact = 1.0
            elif active_crops and any(
                ent == crop + kw for crop in active_crops
            ):
                exact = 0.95
            elif len(kw) > 2 and kw in ent:
                exact = 0.5 if not is_pure_crop else 0.1

            core_tokens = [
                t for t in query_tokens
                if t not in SYSTEM_CONFIG["CROP_LIST"]
                and t != "病"
                and len(t) >= 1
            ]
            if core_tokens and not any(ct in ent for ct in core_tokens):
                dense *= 0.4
                bm *= 0.1

            base = (
                fw["entity_dense"] * dense
                + fw["entity_bm25"] * bm
                + fw["entity_exact"] * exact
            )
            candidates.append({
                "entity": ent,
                "base": base,
                "exact": exact,
            })

        candidates.sort(key=lambda x: x["base"], reverse=True)
        candidates = candidates[
            :int(SYSTEM_CONFIG.get("ENTITY_CANDIDATE_TOP_M", 12))
        ]
        if not candidates:
            continue

        group_id = len(mention_groups)
        mention_groups.append({
            "mention": kw,
            "mention_type": mention_type,
            "candidates": candidates,
        })

        if SYSTEM_CONFIG.get("USE_TYPED_ENTITY_RERANK", True):
            context = query_text or kw
            for cand_idx, cand in enumerate(candidates):
                ent = cand["entity"]
                all_pairs.append([
                    f"{context} [MENTION] {kw}",
                    ENTITY_PROFILES.get(ent, f"实体：{ent}")
                ])
                pair_slots.append((group_id, cand_idx))

    # -------- Stage 2: ONE CrossEncoder call per query --------
    if SYSTEM_CONFIG.get("USE_TYPED_ENTITY_RERANK", True) and all_pairs:
        raw_scores = reranker_model.predict(
            all_pairs,
            batch_size=int(
                SYSTEM_CONFIG.get("ENTITY_CROSS_BATCH_SIZE", 32)
            ),
            show_progress_bar=False
        )
        for raw_score, (group_id, cand_idx) in zip(
            raw_scores, pair_slots
        ):
            mention_groups[group_id]["candidates"][cand_idx][
                "cross"
            ] = _stable_sigmoid(raw_score)
    else:
        for group in mention_groups:
            for cand in group["candidates"]:
                cand["cross"] = cand["base"]

    selected = {}
    selected_details = {}

    for group in mention_groups:
        kw = group["mention"]
        mention_type = group["mention_type"]
        rescored = []

        for cand in group["candidates"]:
            ent = cand["entity"]
            base = cand["base"]
            exact = cand["exact"]
            cross = cand.get("cross", base)

            type_prior = _mention_type_prior(
                mention_type, ent
            )
            crop_prior = _crop_compatibility_prior(
                active_crops, ent
            )
            prior = 0.5 * (type_prior + crop_prior)

            final = (
                SYSTEM_CONFIG["ENTITY_BASE_WEIGHT"] * base
                + SYSTEM_CONFIG["ENTITY_CROSS_WEIGHT"] * cross
                + SYSTEM_CONFIG["ENTITY_PRIOR_WEIGHT"] * prior
            )
            rescored.append({
                "entity": ent,
                "mention": kw,
                "mention_type": mention_type,
                "base": base,
                "cross": cross,
                "prior": prior,
                "final": final,
                "exact": exact,
            })

        rescored.sort(
            key=lambda x: x["final"],
            reverse=True
        )
        if not rescored:
            continue

        def acceptable(item):
            return (
                item["exact"] >= 0.95
                or item["base"] >= threshold
                or (
                    item["cross"]
                    >= SYSTEM_CONFIG["ENTITY_RESCUE_CROSS_MIN"]
                    and item["prior"] >= 0.75
                )
            )

        accepted = []
        best = rescored[0]
        if acceptable(best):
            accepted.append(best)

        if len(rescored) > 1:
            second = rescored[1]
            if (
                acceptable(second)
                and best["final"] - second["final"]
                <= SYSTEM_CONFIG["ENTITY_SECOND_MARGIN"]
            ):
                accepted.append(second)

        for item in accepted:
            ent = item["entity"]
            if (
                ent not in selected
                or item["final"] > selected[ent]
            ):
                selected[ent] = item["final"]
                selected_details[ent] = item

    ranked = sorted(
        selected.items(),
        key=lambda x: x[1],
        reverse=True
    )
    ranked = ranked[
        :int(SYSTEM_CONFIG.get("ENTITY_MAX_TOTAL", 6))
    ]
    ranked_names = {name for name, _ in ranked}
    selected_details = {
        name: selected_details[name]
        for name in ranked_names
        if name in selected_details
    }

    result = (
        dict(ranked)
        if return_scores
        else [x[0] for x in ranked]
    )
    return (result, selected_details) if return_details else result


def select_graph_traversal_seeds(linked_entities, link_details=None,
                                 candidate_diseases=None, limit=None):
    """Select graph-expansion roots independently from entity-F1 ranking.

    Entity resolution and graph expansion have different objectives:
    - entity output favors disambiguation precision;
    - graph seeds favor query evidence that can reach diagnostic triples.

    Therefore symptom/environment/crop evidence is preferred, while
    downstream disease/chemical entities do not consume the main seed budget.
    """
    limit = int(
        limit
        or SYSTEM_CONFIG.get("GRAPH_QUERY_ENTITY_SEEDS", 4)
    )
    link_details = link_details or {}
    disease_set = set(candidate_diseases or [])

    def labels_for(entity):
        return {
            str(x).lower()
            for x in ENTITY_TYPES.get(entity, set())
        }

    def type_priority(entity):
        labels = labels_for(entity)

        if "symptom" in labels or "feature" in labels:
            return 0
        if (
            "environment" in labels
            or "condition" in labels
        ):
            return 1
        if "crop" in labels:
            return 2
        if "disease" in labels or "pest" in labels:
            return 7
        if (
            "chemical" in labels
            or "medicine" in labels
            or "pesticide" in labels
        ):
            return 8
        return 3

    candidates = []
    for order, entity in enumerate(linked_entities):
        if entity in disease_set:
            continue

        labels = labels_for(entity)
        if (
            SYSTEM_CONFIG.get(
                "GRAPH_SEED_EXCLUDE_DISEASE", True
            )
            and (
                "disease" in labels
                or "pest" in labels
                or is_real_disease(entity)
            )
        ):
            continue
        if (
            SYSTEM_CONFIG.get(
                "GRAPH_SEED_EXCLUDE_CHEMICAL", True
            )
            and (
                "chemical" in labels
                or "medicine" in labels
                or "pesticide" in labels
            )
        ):
            continue

        details = link_details.get(entity, {})
        base = float(details.get("base", 0.0))
        exact = float(details.get("exact", 0.0))

        # Lower tuple sorts first.  Type is the primary graph criterion;
        # original lexical/dense evidence is the tie-breaker.
        candidates.append((
            type_priority(entity),
            -exact,
            -base,
            order,
            entity
        ))

    candidates.sort()
    seeds = [x[-1] for x in candidates[:limit]]

    # Conservative fallback: preserve original linked-entity order.
    if len(seeds) < limit:
        for entity in linked_entities:
            if (
                entity in seeds
                or entity in disease_set
            ):
                continue
            seeds.append(entity)
            if len(seeds) >= limit:
                break

    return seeds[:limit]


def graph_search_adaptive(linked_keywords, max_hop, current_weights, query_text="", resolved_crops=None,
                          intent=None, candidate_diseases=None, entity_scores=None, candidate_scores=None,
                          graph_seed_entities=None):
    paths = []
    seen = set()
    mid_node_targets = defaultdict(set)

    if not linked_keywords:
        return [], mid_node_targets, False

    search_seeds = list(graph_seed_entities or linked_keywords)[:int(SYSTEM_CONFIG.get("GRAPH_QUERY_ENTITY_SEEDS", 4))]
    quota_per_seed = 150

    with graph_driver.session() as session:
        for kw in search_seeds:
            if any(x in kw for x in ["病", "斑", "霉", "腐", "虫", "枯"]):
                seed_nodes = session.run(
                    "MATCH (n) WHERE n.name = $kw RETURN n LIMIT 1",
                    kw=kw
                )
            else:
                seed_nodes = session.run(
                    "MATCH (n) WHERE n.name STARTS WITH $kw RETURN n LIMIT 2",
                    kw=kw
                )

            seed_nodes = list(seed_nodes)
            if not seed_nodes:
                continue

            for s in seed_nodes:
                node = s["n"]
                allowed_rels = list(REL_WEIGHTS.keys())

                # Preserve relation direction: retrieve paths that follow KG semantics
                # either away from or towards the query seed; do not use undirected walks.
                result = session.run(f"""
                    MATCH p=(n)-[r*1..{max_hop}]->(m)
                    WHERE elementId(n) = $eid AND n <> m
                      AND all(rel IN relationships(p) WHERE type(rel) IN $allowed_rels)
                    RETURN p
                    UNION
                    MATCH p=(n)<-[r*1..{max_hop}]-(m)
                    WHERE elementId(n) = $eid AND n <> m
                      AND all(rel IN relationships(p) WHERE type(rel) IN $allowed_rels)
                    RETURN p
                    LIMIT {quota_per_seed}
                """,
                eid=node.element_id,
                allowed_rels=allowed_rels)

                for r in result:
                    p = r["p"]
                    if len(p.relationships) > 3:
                        continue

                    nodes = [n["name"] for n in p.nodes]
                    rels = [rel.type for rel in p.relationships]

                    # Crop/category membership is useful as a retrieval filter,
                    # but it is not diagnostic evidence.  In particular, paths
                    # like Disease -> Crop -> Disease only say that two diseases
                    # share a crop; they must not enter PPR, KG-text fusion, or
                    # the displayed reasoning chain.
                    structural_relations = {"belongs_to_crop", "has_disease", "belongs_to_category"}
                    if rels and all(rel in structural_relations for rel in rels):
                        continue

                    if resolved_crops:
                        has_wrong_crop = False
                        for name in nodes:
                            if name in SYSTEM_CONFIG["CROP_LIST"] and name not in resolved_crops:
                                has_wrong_crop = True
                                break
                        if has_wrong_crop:
                            continue

                    path_str = " → ".join(
                        [f"{nodes[i]} → {rels[i]}" for i in range(len(rels))] + [nodes[-1]]
                    )

                    if path_str not in seen:
                        seen.add(path_str)
                        paths.append((nodes, rels, path_str))
                        # A middle node branching to many endpoints is topologically ambiguous.
                        for mid_node in nodes[1:-1]:
                            mid_node_targets[mid_node].add(nodes[-1])

    if not paths:
        return [], mid_node_targets, False

    G = nx.DiGraph()
    path_disease_counts = defaultdict(int)
    for nodes, _, _ in paths:
        detected_ds = [n for n in nodes if is_real_disease(n)]
        for d in set(detected_ds):
            path_disease_counts[d] += 1

    for nodes, rels, _ in paths:
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            rel_type = rels[i]
            edge_w = current_weights.get(rel_type, REL_WEIGHTS.get(rel_type, 1.0))

            if intent and intent in INTENT_RELATION_BOOST:
                if rel_type in INTENT_RELATION_BOOST[intent]:
                    edge_w *= INTENT_RELATION_BOOST[intent][rel_type]

            if G.has_edge(u, v):
                G[u][v]["weight"] += edge_w
            else:
                G.add_edge(u, v, weight=edge_w)

    # Evidence-weighted PPR: entity-linking and diagnosis evidence determine seed mass.
    entity_scores = entity_scores or {}
    candidate_scores = candidate_scores or {}
    personalization = {n: 0.0 for n in G}
    for n in linked_keywords:
        if n in G:
            personalization[n] += max(0.0, float(entity_scores.get(n, 1.0)))
    for disease in (candidate_diseases or []):
        if disease in G:
            personalization[disease] += (SYSTEM_CONFIG["PPR_DIAGNOSIS_LAMBDA"] *
                                          max(0.0, float(candidate_scores.get(disease, 0.0))))
    if not any(personalization.values()):
        personalization = None
    try:
        ppr_scores = nx.pagerank(G, alpha=SYSTEM_CONFIG["PPR_ALPHA"], weight="weight",
                                 personalization=personalization)
    except:
        ppr_scores = {n: 1.0 for n in G.nodes}

    scored_paths = []
    for nodes, rels, path_str in paths:
        score = sum(ppr_scores.get(n, 0) for n in nodes)
        
        # Differential candidates may be softly promoted, never used to exclude alternatives.
        if candidate_diseases and any(n in candidate_diseases for n in nodes):
            score *= 1.10
                
        scored_paths.append((score, path_str))

    scored_paths.sort(reverse=True)
    final_refined = [p for _, p in scored_paths[:SYSTEM_CONFIG["GRAPH_FINAL_TOPK"]]]

    return final_refined, mid_node_targets, True


def decoupled_rerank_system(query, vec_results, kg_paths, reranker, top_n=5):
    if not vec_results:
        return []
        
    pairs = [[query, item["doc"]] for item in vec_results]
    rerank_scores = reranker.predict(pairs)
    
    scored_items = []
    for idx, raw_score in enumerate(rerank_scores):
        item = vec_results[idx]
        norm_rerank_score = 1 / (1 + float(torch.exp(torch.tensor(-raw_score)))) if isinstance(raw_score, float) else float(raw_score)
        
        kg_score = 0.0
        if kg_paths:
            for path in kg_paths[:5]:
                clean_path = path.replace(" → ", " ")
                nodes = [n.strip() for n in clean_path.split() if len(n.strip()) > 1]
                if any(node in item["doc"] for node in nodes):
                    kg_score += 0.2
                    
        kg_score = min(1.0, kg_score)
        fw = VALIDATION_CONFIG["fusion_weights"]
        item["final_composite_score"] = fw["reranker"] * norm_rerank_score + fw["kg"] * kg_score
        scored_items.append(item)
        
    scored_items.sort(key=lambda x: x["final_composite_score"], reverse=True)
    print(f"🔥 解耦二阶段重排精选完毕，顶层复合最终相关分: {scored_items[0]['final_composite_score']:.4f}")
    return scored_items[:top_n]


def score_local_paths(paths, linked_keywords, candidate_diseases=None,
                      entity_scores=None, candidate_scores=None, intent=None):
    """Run the original PPR scoring tail on already retrieved local paths.

    This entry point skips Neo4j traversal; it is used only by the offline demo.
    Each path is (ordered node names, relation names, formatted path string).
    """
    if not paths:
        return [], defaultdict(set), False
    current_weights = REL_WEIGHTS
    mid_node_targets = defaultdict(set)
    G = nx.DiGraph()
    path_disease_counts = defaultdict(int)
    for nodes, _, _ in paths:
        detected_ds = [n for n in nodes if is_real_disease(n)]
        for d in set(detected_ds):
            path_disease_counts[d] += 1

    for nodes, rels, _ in paths:
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            rel_type = rels[i]
            edge_w = current_weights.get(rel_type, REL_WEIGHTS.get(rel_type, 1.0))

            if intent and intent in INTENT_RELATION_BOOST:
                if rel_type in INTENT_RELATION_BOOST[intent]:
                    edge_w *= INTENT_RELATION_BOOST[intent][rel_type]

            if G.has_edge(u, v):
                G[u][v]["weight"] += edge_w
            else:
                G.add_edge(u, v, weight=edge_w)

    # Evidence-weighted PPR: entity-linking and diagnosis evidence determine seed mass.
    entity_scores = entity_scores or {}
    candidate_scores = candidate_scores or {}
    personalization = {n: 0.0 for n in G}
    for n in linked_keywords:
        if n in G:
            personalization[n] += max(0.0, float(entity_scores.get(n, 1.0)))
    for disease in (candidate_diseases or []):
        if disease in G:
            personalization[disease] += (SYSTEM_CONFIG["PPR_DIAGNOSIS_LAMBDA"] *
                                          max(0.0, float(candidate_scores.get(disease, 0.0))))
    if not any(personalization.values()):
        personalization = None
    try:
        ppr_scores = nx.pagerank(G, alpha=SYSTEM_CONFIG["PPR_ALPHA"], weight="weight",
                                 personalization=personalization)
    except:
        ppr_scores = {n: 1.0 for n in G.nodes}

    scored_paths = []
    for nodes, rels, path_str in paths:
        score = sum(ppr_scores.get(n, 0) for n in nodes)
        
        # Differential candidates may be softly promoted, never used to exclude alternatives.
        if candidate_diseases and any(n in candidate_diseases for n in nodes):
            score *= 1.10
                
        scored_paths.append((score, path_str))

    scored_paths.sort(reverse=True)
    final_refined = [p for _, p in scored_paths[:SYSTEM_CONFIG["GRAPH_FINAL_TOPK"]]]

    return final_refined, mid_node_targets, True
