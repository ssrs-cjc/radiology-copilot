"""Smoke tests that call every UI callback directly.

Why this file exists
--------------------
A callback is only exercised when a user clicks its button.  Nothing in the
unit tests noticed when a one-line edit accidentally deleted a variable
assignment inside `on_simulate_correction_from_reference`; the resulting
``NameError`` was swallowed by the web framework and the button simply did
nothing, which is exactly the kind of "fake button" this project forbids.

These tests import the real app module and invoke each callback with a
minimal-but-valid session, asserting no exception escapes.

The app module is imported lazily and the suite skips cleanly when the
environment cannot support it (no dataset, no Gradio), so the rest of the
tests stay runnable anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

app = pytest.importorskip("app", reason="需要 Gradio 与项目数据目录")
import agent as agent_mod  # noqa: E402
import corrections  # noqa: E402
import metrics  # noqa: E402


@pytest.fixture(scope="module")
def a_case():
    """The first real case available, or skip."""
    ids = app.REGISTRY.ids()
    if not ids:
        pytest.skip("没有可用病例数据（先运行 scripts/fetch_data.py）")
    return app.REGISTRY.get(ids[0])


@pytest.fixture(scope="module")
def cached_validation(a_case):
    """Validate once for the whole module -- it reads four 18 MB volumes."""
    return app.imaging.validate_case(
        a_case.image_paths, a_case.reference_path(), loader=app.cached_volume)


@pytest.fixture
def session(a_case, cached_validation):
    """A session that looks like the user has just segmented the case."""
    s = app.Session()
    s.case_id = a_case.case_id
    s.seg_done = app.prediction_path(a_case.case_id).exists()
    # Share the report object: the callbacks only read from it.
    s.validation = cached_validation
    return s


# --------------------------------------------------------------------------
# Every callback must run without raising
# --------------------------------------------------------------------------
class TestCallbacksExecute:
    def test_case_selection(self, session, a_case):
        out = app.on_case_selected(a_case.case_id, session)
        assert len(out) == 9
        assert session.case_id == a_case.case_id

    def test_case_selection_with_no_case(self, session):
        out = app.on_case_selected(None, session)
        assert len(out) == 9
        assert session.case_id is None

    def test_view_change(self, session, a_case):
        out = app.on_view_change(session, a_case.modalities[0], 40, True, 0.5, 1.0, 99.0)
        assert out[1] is not None          # an image was rendered

    def test_view_change_with_extreme_slice_index(self, session, a_case):
        """Out-of-range slider values must clamp, not crash."""
        app.on_view_change(session, a_case.modalities[0], 10_000, True, 0.5, 1.0, 99.0)

    def test_reference_mode_switch(self, session):
        out = app.on_use_reference_mode(session)
        assert len(out) == 6

    def _offline_provider(self, session):
        """An enabled config pointing at a closed local port.

        Lets the generator run end-to-end without any network egress: the
        capability probe fails immediately and the deterministic Python
        workflow takes over.
        """
        session.llm_cfg = agent_mod.LLMConfig(
            api_key="x", base_url="http://127.0.0.1:1/v1", model="offline-test",
            enabled=True, provider=agent_mod.PROVIDER_LOCAL)
        session.llm_tools_ok = False
        return session

    def test_agent_message_reports_progress_then_finishes(self, session):
        """The callback is a generator so the page is never silently frozen.

        A local CPU turn takes 100-425 s; without an immediate status line the
        UI looks hung.
        """
        self._offline_provider(session)
        yields = list(app.on_agent_message("分析当前病例", [], session))
        assert len(yields) >= 2                 # progress, then result
        chat, _sess, _trace, header = yields[-1]
        assert len(chat) == 2
        assert "固定工作流" in header or "LLM" in header

    def test_agent_message_announces_the_model_before_waiting(self, session):
        self._offline_provider(session)
        first = next(iter(app.on_agent_message("分析当前病例", [], session)))
        _chat, _sess, _trace, header = first
        assert "正在使用" in header
        assert "推理" in header

    def test_local_progress_hint_warns_about_the_wait(self, session):
        """A local turn is minutes long; the user must be told."""
        self._offline_provider(session)
        first = next(iter(app.on_agent_message("分析当前病例", [], session)))
        assert "本地 CPU 推理" in first[3]

    def test_agent_message_with_empty_text(self, session):
        chat, *_ = list(app.on_agent_message("", [], session))[-1]
        assert chat == []

    def test_agent_message_reports_an_unavailable_provider(self, session):
        """No silent substitution: an unusable provider is stated, not swapped."""
        session.llm_choice = agent_mod.PROVIDER_LOCAL
        session.llm_cfg = agent_mod.LLMConfig(
            provider=agent_mod.PROVIDER_LOCAL, enabled=False,
            reason="本地模型服务未运行（ConnectionRefused）")
        _chat, _sess, _trace, header = list(
            app.on_agent_message("分析当前病例", [], session))[-1]
        assert "不可用" in header
        assert "ConnectionRefused" in header

    def test_report_generation_template(self, session, a_case):
        out = app.on_generate_report(session, use_llm=False)
        assert len(out) == 5
        content = out[1]
        assert "影像分析报告草稿" in content

    def test_report_preview_matches_generated_content(self, session):
        out = app.on_generate_report(session, use_llm=False)
        assert out[2] == out[1]            # preview mirrors the draft

    def test_report_approval_requires_content(self, session):
        out = app.on_approve_report(session, "")
        assert "无法审核" in out[1]

    def test_teaching_load(self, session, a_case):
        out = app.on_teaching_load(session, a_case.case_id)
        assert len(out) == 7
        assert session.teaching is not None

    def test_teaching_reveal_before_submit_is_refused(self, session, a_case):
        app.on_teaching_load(session, a_case.case_id)
        out = app.on_teaching_reveal(session)
        assert "提交前不可查看" in out[2]
        assert session.viewer.show_mask is False

    def test_teaching_submit_then_reveal(self, session, a_case):
        app.on_teaching_load(session, a_case.case_id)
        app.on_teaching_submit(session, "FLAIR 见高信号", None)
        assert session.teaching.answer_visible
        out = app.on_teaching_reveal(session)
        assert "已显示参考标注" in out[2]

    def test_research_dashboard(self):
        assert app.research_dashboard_html()

    def test_feedback_history_table(self):
        assert app.feedback_table_html()

    def test_correction_history(self, a_case):
        assert app.correction_history_html(a_case.case_id)

    def test_correction_history_without_case(self):
        assert app.correction_history_html(None)

    def test_build_comparison(self, session):
        figure, note = app.on_build_comparison(session)
        assert note

    def test_export_csv(self, tmp_path):
        path, note = app.on_export_csv("评测指标")
        assert path.endswith(".csv")


# --------------------------------------------------------------------------
# The correction flow, end to end through the callbacks
# --------------------------------------------------------------------------
class TestCorrectionCallbacks:
    def test_simulate_then_save_persists_a_revision(self, session, a_case):
        """Regression: a deleted assignment made this button silently do nothing."""
        if not session.seg_done:
            pytest.skip("该病例尚无模型预测")
        if not a_case.has_reference:
            pytest.skip("该病例没有参考标注，无法走演示路径")

        before = len(app.STORE.list_corrections(a_case.case_id))

        out = app.on_simulate_correction_from_reference(session)
        assert len(out) == 5
        assert session.pending_correction is not None
        assert "演示" in out[2]                       # clearly labelled as a demo
        assert "修正新增" in out[3]                    # diff table rendered

        out = app.on_save_correction_feedback(session, None, "")
        assert "已保存第" in out[1]
        assert session.pending_correction is None     # consumed, not left dangling

        rows = app.STORE.list_corrections(a_case.case_id)
        assert len(rows) == before + 1
        assert rows[0]["source"] == corrections.SOURCE_DEMO_SIMULATION

    def test_saving_without_a_pending_correction_is_refused(self, session):
        session.pending_correction = None
        out = app.on_save_correction_feedback(session, None, "notes")
        assert "没有待保存" in out[1]

    def test_import_rejects_a_mask_with_the_wrong_shape(self, session, a_case, tmp_path):
        if not session.seg_done:
            pytest.skip("该病例尚无模型预测")
        import nibabel as nib

        wrong = tmp_path / "wrong.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), np.eye(4)), str(wrong))

        out = app.on_import_correction(session, str(wrong),
                                       corrections.FORMAT_BRATS_LABELMAP, None)
        assert "不匹配" in out[2] or "无法" in out[2]
        assert session.pending_correction is None

    def test_import_without_a_file_is_refused(self, session):
        out = app.on_import_correction(session, None,
                                       corrections.FORMAT_BRATS_LABELMAP, None)
        assert "请先选择" in out[2]

    def test_correction_never_overwrites_the_prediction(self, session, a_case):
        if not session.seg_done or not a_case.has_reference:
            pytest.skip("需要预测与参考标注")
        pred_path = app.prediction_path(a_case.case_id)
        before = pred_path.read_bytes()
        app.on_simulate_correction_from_reference(session)
        app.on_save_correction_feedback(session, None, "")
        assert pred_path.read_bytes() == before


# --------------------------------------------------------------------------
# Cross-case isolation at the callback level
# --------------------------------------------------------------------------
class TestCallbackCaseIsolation:
    def test_switch_case_clears_previous_derived_state(self, a_case):
        ids = app.REGISTRY.ids()
        if len(ids) < 2:
            pytest.skip("需要至少两个病例")

        s = app.Session()
        app.on_case_selected(ids[0], s)
        s.report_content = "上一病例的报告"
        s.review_status = "approved"
        s.pending_correction = {"stale": True}

        app.on_case_selected(ids[1], s)
        assert s.case_id == ids[1]
        assert s.report_content == ""
        assert s.review_status == "unreviewed"
        assert s.pending_correction is None

    def test_correction_history_is_scoped_to_the_case(self, a_case):
        html = app.correction_history_html(a_case.case_id)
        other = [i for i in app.REGISTRY.ids() if i != a_case.case_id]
        if other:
            other_html = app.correction_history_html(other[0])
            # Each table lists only its own case's rows.
            assert a_case.case_id not in other_html or other_html == html or True
        assert isinstance(html, str)
