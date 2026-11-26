from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ead_graphrag import core, evaluation
from ead_graphrag.offline import predict_fixture, candidates_from_triples

ROOT = Path(__file__).resolve().parents[1]


class ScientificMechanismTests(unittest.TestCase):
    def setUp(self):
        core.configure()

    def test_final_policy_is_d1(self):
        self.assertEqual(core.SYSTEM_CONFIG["MIN_DIAGNOSIS_SUPPORT"], 2)
        self.assertEqual(core.SYSTEM_CONFIG["DIAGNOSIS_MARGIN"], 0.2)
        self.assertEqual(core.VALIDATION_CONFIG["diagnosis_fusion_weights"], {"kg": 0.4, "graph": 0.3, "text": 0.3})

    def test_no_candidate_and_one_supported_entity_withhold(self):
        self.assertTrue(core.assess_diagnosis_uncertainty([])[0])
        candidate = {"disease": "示例甲病", "support": 1, "relative_score": 1.0}
        self.assertTrue(core.assess_diagnosis_uncertainty([candidate])[0])
        candidate["support"] = 2
        self.assertFalse(core.assess_diagnosis_uncertainty([candidate])[0])

    def test_ranking_reversal_does_not_override_structural_withholding(self):
        prior = [{"disease": "示例甲病", "support": 2, "relative_score": 0.55},
                 {"disease": "示例乙病", "support": 2, "relative_score": 0.45}]
        docs = [{"meta": {"disease": "示例乙病"}, "final_composite_score": 1.0}]
        fused = core.rescore_differential_diagnosis(prior, docs, ["示例乙病 → has_symptom → 测试线索"])
        self.assertEqual(fused[0]["disease"], "示例乙病")
        withheld, margin = core.assess_diagnosis_uncertainty(fused)
        self.assertTrue(withheld)
        self.assertAlmostEqual(margin, 0.1)

    def test_ppr_receives_query_and_candidate_restart_mass(self):
        local = [(["测试线索", "示例甲病"], ["symptom_of"], "测试线索 → symptom_of → 示例甲病")]
        with patch.object(core.nx, "pagerank", return_value={"测试线索": 0.3, "示例甲病": 0.7}) as pagerank:
            core.score_local_paths(local, ["测试线索", "示例甲病"], ["示例甲病"], {"测试线索": 0.7}, {"示例甲病": 0.8})
        kwargs = pagerank.call_args.kwargs
        self.assertEqual(kwargs["alpha"], 0.85)
        self.assertAlmostEqual(kwargs["personalization"]["测试线索"], 0.7)
        self.assertAlmostEqual(kwargs["personalization"]["示例甲病"], 1.8)

    def test_fixture_labels_cannot_enter_inference(self):
        with self.assertRaises(ValueError):
            predict_fixture({"question": "测试", "gold_diseases": ["答案"]}, [], [])

    def test_all_eight_fictional_cases(self):
        folder = ROOT / "data/synthetic"
        def read(name):
            return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]
        triples, chunks, cases = read("graph.jsonl"), read("chunks.jsonl"), read("cases.jsonl")
        self.assertEqual(len(cases), 8)
        for case in cases:
            inputs = {k: case[k] for k in ("question", "crop", "linked_entities", "entity_scores", "fixture_crossencoder_scores")}
            with self.subTest(case=case["question_id"]), redirect_stdout(io.StringIO()):
                result = predict_fixture(inputs, triples, chunks)
                self.assertEqual(result["uncertain"], case["requires_clarification"])
                if not result["uncertain"]:
                    self.assertIn(result["candidate_diseases"][0], case["gold_diseases"])

    def test_crop_filter_rejects_incompatible_disease(self):
        graph = [{"head": "线索", "relation": "symptom_of", "tail": "示例甲病"},
                 {"head": "示例甲病", "relation": "belongs_to_crop", "tail": "番茄"}]
        self.assertEqual(candidates_from_triples(["线索"], graph, ["黄瓜"]), [])


class EvaluationTests(unittest.TestCase):
    def test_missing_predictions_remain_in_the_denominator(self):
        gold = [{"question_id": str(i), "question": f"测试{i}", "gold_diseases": ["示例甲病"]} for i in range(2)]
        pred = [{"question_id": "0", "candidate_diseases": ["示例甲病"]}]
        scores = evaluation.score(gold, pred)
        self.assertEqual(len(scores), 2)
        self.assertEqual(sum(row["Acc@1"] for row in scores) / len(scores), 0.5)
        self.assertTrue(scores[1]["failed"])

    def test_directed_triple_endpoint_and_relation_are_exact(self):
        gold = [{"question_id": "1", "question": "测试", "gold_kg_triples": [["甲", "symptom_of", "乙"]]}]
        wrong = [{"question_id": "1", "kg_paths": ["乙 → symptom_of → 甲"]}]
        right = [{"question_id": "1", "kg_paths": ["甲 → symptom_of → 乙"]}]
        self.assertEqual(evaluation.score(gold, wrong)[0]["Triple Recall@10"], 0)
        self.assertEqual(evaluation.score(gold, right)[0]["Triple Recall@10"], 1)

    def test_duplicates_and_unexpected_predictions_are_rejected(self):
        gold = [{"question_id": "1", "question": "测试"}]
        with self.assertRaises(ValueError):
            evaluation.score(gold, [{"question_id": "1"}, {"question_id": "1"}])
        with self.assertRaises(ValueError):
            evaluation.score(gold, [{"question_id": "2"}])


if __name__ == "__main__":
    unittest.main()
