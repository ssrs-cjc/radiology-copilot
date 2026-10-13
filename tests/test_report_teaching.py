"""Report consistency and teaching-mode answer withholding."""

from __future__ import annotations

import json

import numpy as np
import pytest

import metrics
import report as report_mod
import teaching as teaching_mod


def make_ctx(**overrides):
    lm = np.zeros((6, 6, 6), dtype=np.uint8)
    lm[1:5, 1:5, 1:5] = 2
    lm[2:4, 2:4, 2:4] = 1
    lm[3, 3, 3] = 4
    volumes = metrics.all_region_volumes(metrics.region_masks(lm, metrics.BRATS_2020),
                                         (1.0, 1.0, 2.0))
    defaults = dict(
        case_id="TEST_001", dataset_source="BraTS 2020", data_origin_label="公开数据集真实病例",
        modalities=["t1", "t1ce", "t2", "flair"],
        analysis_method="MONAI SegResNet 滑窗推理", segmentation_source="AI 模型预测",
        model_provenance="MONAI/brats_mri_segmentation v0.5.4 on cuda:1",
        volumes=volumes, spacing=(1.0, 1.0, 2.0), shape=(6, 6, 6),
        input_checks="检查通过", evaluation=None, review_status="unreviewed",
    )
    defaults.update(overrides)
    return report_mod.ReportContext(**defaults)


# --------------------------------------------------------------------------
# Report consistency - every number must trace to the computed value
# --------------------------------------------------------------------------
class TestReportConsistency:
    def test_report_contains_the_actual_computed_volumes(self):
        ctx = make_ctx()
        text = report_mod.render_template(ctx)
        for region in metrics.REGIONS:
            assert f"{ctx.volumes[region].volume_ml:.2f} mL" in text

    def test_voxel_volume_of_2mm_slice_is_doubled(self):
        ctx = make_ctx()
        # 1x1x2 mm voxel = 0.002 mL; the oedema shell is 5^3 - 3^3 + ... voxels
        assert ctx.volumes["WT"].voxel_volume_ml == pytest.approx(0.002)
        assert ctx.volumes["WT"].volume_ml == pytest.approx(ctx.volumes["WT"].voxel_count * 0.002)

    def test_report_always_carries_the_disclaimer(self):
        text = report_mod.render_template(make_ctx())
        assert report_mod.DISCLAIMER in text
        assert "科研、教学与算法演示" in text
        assert "临床决策" in text

    def test_report_does_not_infer_pathology(self):
        """A segmentation model cannot know histology; the report must not claim it.

        The report *mentions* clinical concepts - but only inside the sentence
        that explicitly disclaims inferring them.  So we assert on where those
        words appear, not merely that they are absent.
        """
        text = report_mod.render_template(make_ctx())

        # Never asserted anywhere, in any context.
        for forbidden in ("胶质母细胞瘤", "星形细胞瘤", "建议手术", "建议放疗", "预后"):
            assert forbidden not in text, f"报告不应出现诊断性表述: {forbidden}"

        # These may appear, but only inside the disclaimer sentence.
        disclaimer_lines = [ln for ln in text.splitlines() if "分子分型" in ln]
        assert len(disclaimer_lines) == 1, "应有的免责声明句缺失"
        assert "不包含" in disclaimer_lines[0]
        for term in ("分级", "病理类型"):
            assert term in disclaimer_lines[0], f"{term} 只应出现在免责声明句中"

    def test_report_separates_observation_from_inference(self):
        text = report_mod.render_template(make_ctx())
        assert "仅陈述本次分析的实际输出" in text

    def test_report_states_the_segmentation_source(self):
        ai = report_mod.render_template(make_ctx(segmentation_source="AI 模型预测"))
        assert "AI 模型预测" in ai
        ref = report_mod.render_template(
            make_ctx(segmentation_source="数据集参考标注", model_provenance=""))
        assert "数据集参考标注" in ref

    def test_empty_measurements_render_an_honest_placeholder(self):
        text = report_mod.render_template(make_ctx(volumes={}))
        assert "尚未执行分割" in text
        assert "0.00 mL" not in text          # no fake zeros

    def test_research_metrics_are_absent_from_the_clinical_report(self):
        """Dice and voxel-error counts describe the algorithm, not the patient.

        They must not appear in the imaging report body; the report instead
        points the reader at the research module.
        """
        ev = {"dice": {"WT": 0.91, "TC": 0.88, "ET": 0.80},
              "missed_voxels": {"WT": 10, "TC": 5, "ET": 2},
              "extra_voxels": {"WT": 3, "TC": 1, "ET": 0}}
        text = report_mod.render_template(make_ctx(evaluation=ev))

        assert "0.9100" not in text
        assert "0.8800" not in text
        assert "对比评测" not in text
        assert "漏分割体素" not in text
        # ... but the reader is told where to find them.
        assert "科研反馈" in text

    def test_report_without_evaluation_still_renders_completely(self):
        text = report_mod.render_template(make_ctx(evaluation=None))
        for section in ("检查信息", "定量分析", "影像观察", "分析摘要", "备注"):
            assert section in text
        assert "科研反馈" not in text      # no pointer when there is nothing to point at

    def test_llm_facts_exclude_research_metrics(self):
        """The model cannot write Dice into the narrative if it never sees it."""
        ev = {"dice": {"WT": 0.91, "TC": 0.88, "ET": 0.80},
              "missed_voxels": {"WT": 10}, "extra_voxels": {"WT": 3}}
        facts = report_mod.facts_for_llm(make_ctx(evaluation=ev))
        assert "evaluation" not in facts
        assert "dice" not in json.dumps(facts).lower()

    def test_status_label_reflects_review_state(self):
        assert "已人工审核确认" in report_mod.render_template(make_ctx(review_status="approved"))
        assert "待审核" in report_mod.render_template(make_ctx(review_status="pending_review"))


class TestLlmNarrationGuardrails:
    def test_placeholder_substitution_uses_computed_values(self):
        ctx = make_ctx()
        out = report_mod.substitute_placeholders(
            "WT 为 {{WT_ML}}，TC 为 {{TC_ML}}，ET 为 {{ET_ML}}。", ctx)
        assert "{{" not in out
        assert f"{ctx.volumes['WT'].volume_ml:.2f} mL" in out

    def test_placeholder_substitution_collapses_a_duplicated_unit(self):
        """Models write `{{WT_ML}} mL` even though the placeholder has the unit.

        Regression: that rendered as "192.53 mL mL".
        """
        ctx = make_ctx()
        out = report_mod.substitute_placeholders(
            "全肿瘤区 {{WT_ML}} mL，核心区 {{TC_ML}}mL。", ctx)
        assert "mL mL" not in out
        assert "mLmL" not in out
        assert out.count("mL") == 2

    def test_facts_payload_is_json_serialisable(self):
        import json
        json.dumps(report_mod.facts_for_llm(make_ctx()))

    def test_system_prompt_forbids_fabrication(self):
        prompt = report_mod.LLM_SYSTEM_PROMPT
        assert "不得编造" in prompt
        assert "不得推断" in prompt
        assert "{{WT_ML}}" in prompt

    def test_merge_keeps_template_when_llm_output_is_malformed(self):
        template = report_mod.render_template(make_ctx())
        merged = report_mod.merge_llm_into_template(template, "no headings here", "m")
        assert merged == template

    def test_merge_labels_the_narrative_as_model_generated(self):
        template = report_mod.render_template(make_ctx())
        llm = "### 影像观察\n- 检出 WT。\n\n### 分析摘要\n完成了分割。"
        merged = report_mod.merge_llm_into_template(template, llm, "test-model")
        assert "test-model" in merged
        assert "确定性代码计算并填入" in merged


# --------------------------------------------------------------------------
# Teaching - the answer must not leak before submission
# --------------------------------------------------------------------------
class TestAnswerWithholding:
    def test_reference_hidden_before_submission(self):
        st = teaching_mod.ExerciseState(case_id="c", reference_label_map=np.ones((4, 4, 4)))
        assert st.reference_for_display() is None
        assert not st.answer_visible

    def test_reference_available_after_submission(self):
        st = teaching_mod.ExerciseState(case_id="c", reference_label_map=np.ones((4, 4, 4)))
        st.submitted = True
        assert st.reference_for_display() is not None
        assert st.answer_visible

    def test_reset_hides_the_answer_again(self):
        st = teaching_mod.ExerciseState(case_id="c", reference_label_map=np.ones((4, 4, 4)))
        st.submitted = True
        st.reset()
        assert st.reference_for_display() is None


class TestMaskScoring:
    def test_perfect_submission_scores_one(self, brats_label_map):
        ev = teaching_mod.score_mask_submission(
            brats_label_map, brats_label_map, (1.0, 1.0, 1.0), metrics.BRATS_2020)
        assert ev["comparable"] and ev["mean_dice"] == pytest.approx(1.0)

    def test_partial_submission_reports_missed_and_extra(self, brats_label_map):
        pred = brats_label_map.copy()
        pred[0:2, 0:2, :] = 4          # spurious enhancing region
        pred[pred == 2] = 0            # drop oedema
        ev = teaching_mod.score_mask_submission(
            pred, brats_label_map, (1.0, 1.0, 1.0), metrics.BRATS_2020)
        assert ev["regions"]["WT"]["missed_voxels"] > 0
        assert ev["regions"]["WT"]["extra_voxels"] > 0

    def test_shape_mismatch_is_reported_not_scored(self, brats_label_map):
        ev = teaching_mod.score_mask_submission(
            np.zeros((2, 2, 2), np.uint8), brats_label_map, (1, 1, 1), metrics.BRATS_2020)
        assert ev["comparable"] is False
        assert "尺寸" in ev["reason"]

    def test_volume_delta_is_signed(self, brats_label_map):
        pred = brats_label_map.copy()
        pred[pred == 2] = 0
        ev = teaching_mod.score_mask_submission(
            pred, brats_label_map, (1.0, 1.0, 1.0), metrics.BRATS_2020)
        assert ev["regions"]["WT"]["volume_delta_ml"] < 0


class TestTextAnswerReview:
    def test_no_fake_score_is_produced(self):
        review = teaching_mod.review_text_answer(
            "FLAIR 序列见大片高信号，考虑胶质母细胞瘤", {"modalities": ["flair", "t1ce"]})
        assert "score" not in review
        assert "matched_terms" in review        # a checklist count, not a grade

    def test_clinical_judgement_is_explicitly_not_graded(self):
        review = teaching_mod.review_text_answer(
            "考虑胶质母细胞瘤 WHO IV 级", {"modalities": ["flair"]})
        assert "不进行自动评分" in review["not_graded"]
        assert any("诊断" in c["item"] for c in review["checklist"])

    def test_terminology_checklist_reflects_available_modalities(self):
        review = teaching_mod.review_text_answer(
            "T1ce 增强扫描见环形强化", {"modalities": ["t1ce", "flair"]})
        hits = {c["item"]: c["status"] for c in review["checklist"]}
        assert hits["提及 T1CE 序列"] == "hit"
        assert hits["提及 FLAIR 序列"] == "miss"

    def test_empty_answer_returns_empty_checklist(self):
        review = teaching_mod.review_text_answer("", {"modalities": ["t1"]})
        assert review["checklist"] == []


class TestFeedbackFormatting:
    def test_feedback_states_dice_is_not_clinical_accuracy(self):
        ev = teaching_mod.score_mask_submission(
            np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
            (1, 1, 1), metrics.BRATS_2020)
        md = teaching_mod.feedback_markdown(ev, None, True)
        assert "不进行自动评分" in md

    def test_missing_reference_is_stated_plainly(self):
        md = teaching_mod.feedback_markdown({}, None, reference_available=False)
        assert "无法计算分割精度指标" in md
