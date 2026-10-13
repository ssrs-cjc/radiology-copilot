"""Agent tool behaviour and persistence of review / feedback records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import agent as agent_mod
import cases as cases_mod
import metrics
import research
import segmentation
import storage


# --------------------------------------------------------------------------
# Agent registry: every tool reports a status, and gated tools refuse
# --------------------------------------------------------------------------
@pytest.fixture
def bare_session(tmp_path):
    """A session with a case but no analysis at all."""
    case = cases_mod.Case(
        case_id="TEST_001", dataset_source="BraTS 2020", modalities=("t1", "t1ce", "t2", "flair"),
        image_paths={}, label_convention="brats_2020",
        data_origin=cases_mod.ORIGIN_PUBLIC)
    return agent_mod.AgentSession(case=case, store=storage.Store(tmp_path / "t.db"), case_id="TEST_001")


@pytest.fixture
def analysed_session(bare_session):
    """A session with a real prediction, reference and evaluation attached."""
    lm = np.zeros((8, 8, 8), dtype=np.uint8)
    lm[2:6, 2:6, 2:6] = 2
    lm[3:5, 3:5, 3:5] = 1
    lm[3, 3, 3] = 4
    ref = lm.copy()
    pred = lm.copy()
    pred[pred == 2] = 0                      # deliberately under-segmented

    bare_session.segmentation = segmentation.SegmentationResult(
        label_map=pred, region_masks=metrics.region_masks(pred, metrics.BRATS_2020),
        channel_probabilities=None, spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4), duration_s=4.2, device="cuda:1")
    bare_session.reference = segmentation.ReferenceAnnotation(
        label_map=ref, region_masks=metrics.region_masks(ref, metrics.BRATS_2020),
        spacing=(1.0, 1.0, 1.0), affine=np.eye(4), source="test reference")
    bare_session.evaluation = metrics.evaluate_segmentation(pred, ref, (1.0, 1.0, 1.0))
    return bare_session


class TestToolRegistration:
    def test_expected_tools_are_registered(self, bare_session):
        names = set(agent_mod.build_registry(bare_session).names())
        assert {"get_case_info", "validate_mri", "run_segmentation",
                "get_segmentation_results", "calculate_volume", "get_review_status",
                "get_model_evaluation", "generate_report_draft",
                "get_research_feedback"} <= names

    def test_no_shell_or_file_tools_are_exposed(self, bare_session):
        """The agent must not be able to run commands or touch the filesystem."""
        names = " ".join(agent_mod.build_registry(bare_session).names()).lower()
        for forbidden in ("shell", "exec", "bash", "write_file", "delete", "rm_"):
            assert forbidden not in names

    def test_schemas_are_valid_openai_function_specs(self, bare_session):
        for schema in agent_mod.build_registry(bare_session).schemas():
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    def test_unknown_tool_is_refused_not_executed(self, bare_session):
        reg = agent_mod.build_registry(bare_session)
        result = reg.invoke("drop_database", {})
        assert result.status == "refused"
        assert "未注册" in result.summary

    def test_every_invocation_is_recorded_in_the_trace(self, bare_session):
        reg = agent_mod.build_registry(bare_session)
        reg.invoke("get_case_info", {})
        reg.invoke("not_a_tool", {})
        assert len(reg.calls) == 2
        assert {c["tool"] for c in reg.calls} == {"get_case_info", "not_a_tool"}


class TestToolStatuses:
    def test_segmentation_tool_refuses_without_user_confirmation(self, bare_session):
        """The assistant may never start a model run on its own."""
        reg = agent_mod.build_registry(bare_session)
        result = reg.invoke("run_segmentation", {"confirm": False})
        assert result.status == "refused"
        assert "confirm" in result.error or "确认" in result.summary

    def test_segmentation_tool_refuses_even_when_confirmed_by_the_model(self, bare_session):
        """Confirmation must come from the UI, not from the model's own argument."""
        reg = agent_mod.build_registry(bare_session)
        result = reg.invoke("run_segmentation", {"confirm": True})
        assert result.status == "refused"
        assert "界面" in result.summary

    def test_missing_segmentation_reports_error_not_fabrication(self, bare_session):
        result = agent_mod.build_registry(bare_session).invoke("get_segmentation_results", {})
        assert result.status == "error"
        assert result.data == {}                 # no invented numbers

    def test_volume_tool_reports_error_when_no_mask_exists(self, bare_session):
        result = agent_mod.build_registry(bare_session).invoke("calculate_volume", {})
        assert result.status == "error"

    def test_volume_tool_returns_deterministic_values(self, analysed_session):
        result = agent_mod.build_registry(analysed_session).invoke("calculate_volume", {})
        assert result.status == "ok"
        assert result.data["computed_by"].startswith("deterministic")
        assert result.data["source"] == "model_prediction"
        # The tool reports the *prediction*, which this fixture deliberately
        # under-segments by deleting every oedema voxel.  What remains is the
        # 2x2x2 core block (8 voxels, one of which was overwritten by label 4):
        # 7 voxels of label 1 plus 1 voxel of label 4 = 8 voxels at 1 mm^3.
        assert result.data["WT"] == pytest.approx(8 * 0.001)
        assert result.data["TC"] == pytest.approx(8 * 0.001)
        assert result.data["ET"] == pytest.approx(1 * 0.001)
        # ... and it must be strictly smaller than the reference, proving the
        # numbers come from the mask rather than from the reference.
        assert result.data["WT"] < 64 * 0.001

    def test_reference_only_session_is_flagged_as_not_a_prediction(self, bare_session):
        lm = np.zeros((4, 4, 4), np.uint8); lm[1:3, 1:3, 1:3] = 2
        bare_session.reference = segmentation.ReferenceAnnotation(
            label_map=lm, region_masks=metrics.region_masks(lm, metrics.BRATS_2020),
            spacing=(1, 1, 1), affine=np.eye(4), source="ref")
        result = agent_mod.build_registry(bare_session).invoke("get_segmentation_results", {})
        assert result.status == "ok"
        assert result.data["is_model_prediction"] is False
        assert "参考标注" in result.summary

    def test_evaluation_tool_explains_missing_reference(self, bare_session):
        result = agent_mod.build_registry(bare_session).invoke("get_model_evaluation", {})
        assert result.data["available"] is False
        assert "参考标注" in result.summary or "Dice" in result.summary

    def test_evaluation_tool_returns_real_dice(self, analysed_session):
        result = agent_mod.build_registry(analysed_session).invoke("get_model_evaluation", {})
        assert result.status == "ok"
        assert result.data["available"] is True
        assert result.data["dice"]["WT"] < 1.0     # the fixture is under-segmented

    def test_case_info_does_not_run_an_analysis(self, bare_session):
        result = agent_mod.build_registry(bare_session).invoke("get_case_info", {})
        assert result.status == "ok"
        assert result.data["segmentation_status"] == "not_run"

    def test_tool_data_is_json_serialisable(self, analysed_session):
        reg = agent_mod.build_registry(analysed_session)
        for name in reg.names():
            args = {"confirm": True} if name == "run_segmentation" else {}
            json.dumps(reg.invoke(name, args).data, default=str)


class TestFallbackWorkflow:
    def test_no_api_key_uses_fixed_workflow_and_says_so(self, bare_session):
        cfg = agent_mod.LLMConfig(reason="test: no key")
        turn = agent_mod.run_agent_turn("分析当前病例", bare_session, cfg)
        assert turn.mode == "fixed_workflow"
        assert "固定工作流模式" in turn.answer

    def test_fallback_actually_executes_tools(self, analysed_session):
        cfg = agent_mod.LLMConfig(reason="test")
        turn = agent_mod.run_agent_turn("显示肿瘤区域体积", analysed_session, cfg)
        executed = [s["tool"] for s in turn.tool_calls]
        assert "calculate_volume" in executed
        assert all(s["status"] in {"ok", "error", "refused"} for s in turn.tool_calls)

    def test_fallback_answer_quotes_real_numbers(self, analysed_session):
        cfg = agent_mod.LLMConfig(reason="test")
        turn = agent_mod.run_agent_turn("显示肿瘤区域体积", analysed_session, cfg)
        assert "mL" in turn.answer
        assert "确定性代码" in turn.answer

    def test_probe_failure_falls_back(self, bare_session):
        """Even with a key configured, a provider without tool support must not
        be allowed to narrate tool calls it cannot make."""
        cfg = agent_mod.LLMConfig(api_key="x", model="m", enabled=True)
        turn = agent_mod.run_agent_turn("分析当前病例", bare_session, cfg, tools_supported=False)
        assert turn.mode == "fixed_workflow"
        assert "未通过函数调用能力探测" in turn.answer

    def test_trace_renders_executed_tools(self, analysed_session):
        cfg = agent_mod.LLMConfig(reason="test")
        turn = agent_mod.run_agent_turn("分析当前病例", analysed_session, cfg)
        md = turn.trace_markdown()
        assert "get_case_info" in md

    def test_empty_message_is_handled(self, bare_session):
        cfg = agent_mod.LLMConfig(reason="test")
        turn = agent_mod.run_agent_turn("", bare_session, cfg)
        assert turn.answer          # must not crash or return None


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
class TestStorage:
    def test_case_round_trip_preserves_json_fields(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds", modalities=["t1", "t2"],
                       image_paths={"t1": "a.nii"}, measurements={"WT": {"ml": 1.5}})
        rec = st.get_case("C1")
        assert rec["modalities"] == ["t1", "t2"]
        assert rec["measurements"]["WT"]["ml"] == 1.5

    def test_reports_are_append_only_revisions(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds")
        st.save_report("C1", "v1", "pending_review", "template")
        st.save_report("C1", "v2", "approved", "human", approve=True)
        history = st.report_history("C1")
        assert [h["revision"] for h in history] == [2, 1]
        assert st.latest_report("C1")["content"] == "v2"

    def test_approving_a_report_does_not_delete_the_prediction(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds", prediction_mask_path="pred.nii.gz")
        st.save_report("C1", "content", "approved", "human", approve=True)
        assert st.get_case("C1")["prediction_mask_path"] == "pred.nii.gz"
        assert st.get_case("C1")["review_status"] == "approved"

    def test_feedback_persists_and_aggregates(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds")
        st.record_feedback("C1", "under_segmentation", model_name="m", model_version="1",
                           notes="note")
        st.record_feedback("C1", "under_segmentation", model_name="m")
        st.record_feedback("C1", "boundary_deviation", model_name="m")
        assert st.feedback_counts() == {"under_segmentation": 2, "boundary_deviation": 1}
        assert len(st.list_feedback()) == 3

    def test_analysis_runs_record_model_version_and_device(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds")
        st.record_analysis_run("C1", "MONAI/brats", "success", model_version="0.5.4",
                               device="cuda:1", duration_s=4.0,
                               evaluation={"dice": {"WT": 0.9, "TC": 0.8, "ET": 0.7}})
        runs = st.list_analysis_runs("C1")
        assert runs[0]["model_version"] == "0.5.4"
        assert runs[0]["device"] == "cuda:1"

    def test_failed_runs_are_recorded_too(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds")
        st.record_analysis_run("C1", "MONAI/brats", "failed", error_message="boom")
        assert st.list_analysis_runs("C1")[0]["status"] == "failed"
        assert st.counts()["analyses"] == 0        # failures are not counted as analyses

    def test_agent_trace_records_real_tool_calls(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.record_agent_trace("C1", "分析", "fixed_workflow",
                              [{"tool": "get_case_info", "status": "ok"}], "answer", "m")
        trace = st.list_agent_traces()[0]
        assert trace["tool_calls"][0]["tool"] == "get_case_info"
        assert trace["mode"] == "fixed_workflow"


class TestResearchAggregation:
    def _seed(self, st, dice_values):
        st.upsert_case("C1", dataset_source="ds")
        for i, (wt, tc, et) in enumerate(dice_values):
            st.record_analysis_run(
                f"C1", "MONAI/brats_mri_segmentation", "success", model_version="0.5.4",
                evaluation={"dice": {"WT": wt, "TC": tc, "ET": et},
                            "missed_voxels": {"WT": 1}, "extra_voxels": {"WT": 0}})

    def test_empty_database_shows_an_empty_state_not_zeros(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        stats = research.aggregate(st)
        assert not stats.has_data
        # Pass an explicit empty batch so the test does not depend on whether a
        # batch evaluation file happens to exist on this machine.
        md = research.dashboard_markdown(stats, research.BatchEvaluation(available=False))
        assert "暂无科研评测数据" in md
        assert "0.000" not in md

    def test_means_are_computed_only_from_evaluated_cases(self, tmp_path):
        """Two *distinct* cases; a case scored twice is still one case."""
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds")
        st.upsert_case("B", dataset_source="ds")
        for cid, wt in (("A", 0.9), ("B", 0.7)):
            st.record_analysis_run(cid, "MONAI/brats_mri_segmentation", "success",
                                   model_version="0.5.4",
                                   evaluation={"dice": {"WT": wt, "TC": wt, "ET": wt},
                                               "missed_voxels": {"WT": 1}, "extra_voxels": {"WT": 0}})
        stats = research.aggregate(st)
        assert stats.n_evaluated_cases == 2
        assert stats.mean_dice["WT"] == pytest.approx(0.8)

    def test_cases_without_evaluation_do_not_drag_the_mean_down(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        self._seed(st, [(0.9, 0.9, 0.9)])
        st.record_analysis_run("C1", "MONAI/brats_mri_segmentation", "success",
                               model_version="0.5.4", evaluation={})
        stats = research.aggregate(st)
        assert stats.n_evaluated_cases == 1
        assert stats.mean_dice["WT"] == pytest.approx(0.9)

    def test_hard_cases_are_ranked_by_whole_tumour_dice(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        for cid, wt in (("A", 0.9), ("B", 0.35), ("C", 0.55)):
            st.upsert_case(cid, dataset_source="ds")
            st.record_analysis_run(cid, "MONAI/brats", "success", model_version="0.5.4",
                                   evaluation={"dice": {"WT": wt, "TC": wt, "ET": wt},
                                               "missed_voxels": {}, "extra_voxels": {}})
        stats = research.aggregate(st)
        assert [c["case_id"] for c in stats.hard_cases] == ["B", "C"]

    def test_csv_export_has_bom_and_real_rows(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("C1", dataset_source="ds")
        st.record_feedback("C1", "under_segmentation", model_name="m", notes="n")
        csv_text = research.feedback_csv(st.list_feedback())
        assert csv_text.startswith("﻿")           # Excel-friendly
        assert "漏分割" in csv_text                     # Chinese label, not the key
        assert "under_segmentation" not in csv_text

    def test_evaluation_csv_contains_dice_columns(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        self._seed(st, [(0.9, 0.8, 0.7)])
        csv_text = research.evaluation_csv(st)
        header = csv_text.splitlines()[0]
        assert "WT Dice" in header and "TC Dice" in header and "ET Dice" in header
        assert "0.9000" in csv_text

    def test_error_type_suggestion_matches_the_error_direction(self):
        assert research.suggest_error_type({"WT": 500}, {"WT": 0}) == "under_segmentation"
        assert research.suggest_error_type({"WT": 0}, {"WT": 500}) == "over_segmentation"
        assert research.suggest_error_type({"WT": 100}, {"WT": 100}) == "boundary_deviation"
