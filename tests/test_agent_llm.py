"""LLM Agent behaviour: argument validation, real tool execution, fallback.

A fake OpenAI client drives the *real* tool-calling loop, so these tests
exercise the actual control flow (schema validation, execution, tracing,
consistency checking) without needing a network or an API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import agent as agent_mod
import cases as cases_mod
import metrics
import report as report_mod
import segmentation
import storage


# --------------------------------------------------------------------------
# Minimal OpenAI-compatible fakes
# --------------------------------------------------------------------------
class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeResponse:
    def __init__(self, message: _FakeMessage):
        self.choices = [SimpleNamespace(message=message)]


class FakeOpenAI:
    """Replays a scripted list of assistant messages, recording each request."""

    def __init__(self, script: list[_FakeMessage] | None = None, **_kwargs):
        self._script = list(script or [])
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._script:
            return _FakeResponse(_FakeMessage(content="（脚本已用尽）"))
        return _FakeResponse(self._script.pop(0))


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a scripted fake client and return a helper to set the script."""
    holder: dict[str, FakeOpenAI] = {}

    def _install(script: list[_FakeMessage]) -> FakeOpenAI:
        fake = FakeOpenAI(script)
        holder["client"] = fake
        monkeypatch.setattr("openai.OpenAI", lambda **kw: fake, raising=True)
        return fake

    _install.holder = holder  # type: ignore[attr-defined]
    return _install


def tool_call(call_id: str, name: str, arguments: dict) -> _FakeToolCall:
    return _FakeToolCall(call_id, name, json.dumps(arguments))


CFG = agent_mod.LLMConfig(api_key="test-key-not-real", base_url="http://fake",
                          model="fake-model", enabled=True)


# --------------------------------------------------------------------------
# Session fixtures
# --------------------------------------------------------------------------
def _prediction() -> np.ndarray:
    lm = np.zeros((8, 8, 8), dtype=np.uint8)
    lm[2:6, 2:6, 2:6] = 2          # oedema -> WT
    lm[3:5, 3:5, 3:5] = 1          # core   -> WT + TC
    lm[3, 3, 3] = 4                # enhancing -> WT + TC + ET
    return lm


@pytest.fixture
def analysed_session(tmp_path):
    """A session with a real prediction, reference and evaluation attached."""
    case = cases_mod.Case(
        case_id="BraTS20_Training_001", dataset_source="BraTS 2020",
        modalities=("t1", "t1ce", "t2", "flair"), image_paths={},
        label_convention="brats_2020", data_origin=cases_mod.ORIGIN_PUBLIC)
    pred = _prediction()
    ref = pred.copy()
    ref[ref == 2] = 0              # reference has no oedema -> under-segmented

    session = agent_mod.AgentSession(
        case=case, case_id=case.case_id,
        store=storage.Store(tmp_path / "t.db"),
        segmentation=segmentation.SegmentationResult(
            label_map=pred, region_masks=metrics.region_masks(pred, metrics.BRATS_2020),
            channel_probabilities=None, spacing=(1.0, 1.0, 1.0),
            affine=np.eye(4), duration_s=1.0, device="cuda:0"),
        reference=segmentation.ReferenceAnnotation(
            label_map=ref, region_masks=metrics.region_masks(ref, metrics.BRATS_2020),
            spacing=(1.0, 1.0, 1.0), affine=np.eye(4), source="ref"),
        evaluation=metrics.evaluate_segmentation(pred, ref, (1.0, 1.0, 1.0)),
    )
    return session


def _volume_report() -> tuple[str, str]:
    """A stand-in report builder returning a real computed volume."""
    return ("# 影像分析报告草稿\n\n- 全肿瘤区：0.062 mL\n", "template")


# ==========================================================================
# 1. Argument validation
# ==========================================================================
class TestArgumentValidation:
    def test_unknown_property_is_rejected(self):
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}},
                  "required": ["confirm"]}
        with pytest.raises(agent_mod.ToolArgumentError, match="未声明的参数"):
            agent_mod.validate_arguments("run_segmentation", schema,
                                         {"confirm": True, "shell": "rm -rf /"})

    def test_missing_required_is_rejected(self):
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}},
                  "required": ["confirm"]}
        with pytest.raises(agent_mod.ToolArgumentError, match="缺少必需参数"):
            agent_mod.validate_arguments("run_segmentation", schema, {})

    def test_wrong_type_is_rejected(self):
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}},
                  "required": ["confirm"]}
        with pytest.raises(agent_mod.ToolArgumentError, match="期望 boolean"):
            agent_mod.validate_arguments("run_segmentation", schema, {"confirm": "yes"})

    def test_integer_where_boolean_expected_is_rejected(self):
        """bool subclasses int in Python, so this needs an explicit guard."""
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}},
                  "required": []}
        with pytest.raises(agent_mod.ToolArgumentError):
            agent_mod.validate_arguments("t", schema, {"confirm": 1})

    def test_non_object_arguments_rejected(self):
        with pytest.raises(agent_mod.ToolArgumentError, match="JSON 对象"):
            agent_mod.validate_arguments("t", {"type": "object"}, ["not", "a", "dict"])

    def test_valid_arguments_pass_through(self):
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}},
                  "required": ["confirm"]}
        assert agent_mod.validate_arguments("t", schema, {"confirm": True}) == {"confirm": True}

    def test_registry_returns_error_instead_of_raising(self, analysed_session):
        reg = agent_mod.build_registry(analysed_session)
        result = reg.invoke("run_segmentation", {"confirm": True, "bogus": 1})
        assert result.status == "error"
        assert "参数校验未通过" in result.summary
        assert "bogus" in result.error

    def test_enum_is_enforced(self):
        schema = {"type": "object",
                  "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
                  "required": ["mode"]}
        with pytest.raises(agent_mod.ToolArgumentError, match="取值必须是"):
            agent_mod.validate_arguments("t", schema, {"mode": "c"})


# ==========================================================================
# 2. Tool failure and error feedback
# ==========================================================================
class TestToolFailureFeedback:
    def test_tool_exception_becomes_an_error_result(self, analysed_session):
        reg = agent_mod.build_registry(analysed_session)
        reg.register(agent_mod.Tool(
            "explode", "always fails", {"type": "object", "properties": {}, "required": []},
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
        result = reg.invoke("explode", {})
        assert result.status == "error"
        assert "RuntimeError" in result.summary
        assert "boom" in result.error

    def test_error_is_returned_to_the_model_not_swallowed(self, analysed_session, fake_llm):
        """The loop must feed the failure back so the model can react."""
        fake = fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "no_such_tool", {})]),
            _FakeMessage(content="该工具不存在。"),
        ])
        turn = agent_mod.run_agent_turn("分析", analysed_session, CFG, tools_supported=True)
        assert turn.tool_calls[0]["status"] == "refused"
        # Second request carries the tool error back to the model.
        tool_msgs = [m for m in fake.requests[1]["messages"] if m.get("role") == "tool"]
        assert tool_msgs and "refused" in tool_msgs[0]["content"]

    def test_tool_error_does_not_abort_the_turn(self, analysed_session, fake_llm):
        fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "calculate_volume", {"bogus": 1})]),
            _FakeMessage(content="参数有误，请重试。"),
        ])
        turn = agent_mod.run_agent_turn("体积", analysed_session, CFG, tools_supported=True)
        assert turn.answer == "参数有误，请重试。"
        assert turn.error == ""


# ==========================================================================
# 3. Fallback when the API is not configured
# ==========================================================================
class TestNoApiFallback:
    def test_missing_key_uses_fixed_workflow_and_says_so(self, analysed_session):
        cfg = agent_mod.LLMConfig(reason="未设置 LLM_API_KEY")
        turn = agent_mod.run_agent_turn("分析当前病例", analysed_session, cfg)
        assert turn.mode == "fixed_workflow"
        assert "未配置语言模型 API" in turn.answer

    def test_missing_key_never_claims_llm_execution(self, analysed_session):
        cfg = agent_mod.LLMConfig(reason="未设置 LLM_API_KEY")
        turn = agent_mod.run_agent_turn("分析当前病例", analysed_session, cfg)
        assert turn.mode != "llm_tool_calling"
        assert turn.model == ""            # no model attributed

    def test_probe_failure_falls_back(self, analysed_session):
        turn = agent_mod.run_agent_turn("分析当前病例", analysed_session, CFG,
                                        tools_supported=False)
        assert turn.mode == "fixed_workflow"
        assert "未通过函数调用能力探测" in turn.answer

    def test_load_config_reports_the_missing_variable(self, monkeypatch):
        for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            monkeypatch.delenv(var, raising=False)
        cfg = agent_mod.load_llm_config()
        assert not cfg.enabled
        assert "LLM_API_KEY" in cfg.reason

    def test_config_never_leaks_the_key_in_its_description(self):
        cfg = agent_mod.LLMConfig(api_key="sk-super-secret", base_url="http://x",
                                  model="m", enabled=True)
        assert "sk-super-secret" not in cfg.describe()


# ==========================================================================
# 4. Illegal tool names
# ==========================================================================
class TestIllegalToolNames:
    def test_unregistered_tool_is_refused_with_the_available_list(self, analysed_session):
        reg = agent_mod.build_registry(analysed_session)
        result = reg.invoke("os_system", {"cmd": "ls"})
        assert result.status == "refused"
        assert "get_case_info" in result.error     # tells the model what it may use

    def test_no_dangerous_tool_names_are_registered(self, analysed_session):
        """No tool may execute a command or touch the filesystem.

        Matched on whole tokens rather than substrings: `get_model_evaluation`
        contains "eval" but is a read-only metric lookup.
        """
        names = agent_mod.build_registry(analysed_session).names()
        tokens = {tok for n in names for tok in n.lower().split("_")}
        for forbidden in ("shell", "exec", "bash", "eval", "system", "sh",
                          "write", "delete", "remove", "rm", "unlink", "open"):
            assert forbidden not in tokens, f"工具名中出现危险动作: {forbidden}"

    def test_registry_rejects_malformed_tool_names(self, analysed_session):
        reg = agent_mod.build_registry(analysed_session)
        with pytest.raises(ValueError):
            reg.register(agent_mod.Tool("bad name!", "d", {"type": "object"},
                                        lambda: None))


# ==========================================================================
# 5. Unauthorised state changes
# ==========================================================================
class TestUnauthorisedStateChange:
    def test_run_segmentation_refused_without_confirmation(self, analysed_session):
        result = agent_mod.build_registry(analysed_session).invoke(
            "run_segmentation", {"confirm": False})
        assert result.status == "refused"

    def test_run_segmentation_refused_even_when_the_model_confirms(self, analysed_session):
        """Confirmation must come from the user, never from the model's own argument."""
        result = agent_mod.build_registry(analysed_session).invoke(
            "run_segmentation", {"confirm": True})
        assert result.status == "refused"
        assert "界面" in result.summary

    def test_no_tool_can_approve_a_report(self, analysed_session):
        """Approval is a human action; it must not be reachable by the agent."""
        names = agent_mod.build_registry(analysed_session).names()
        assert not any("approve" in n or "review" in n and "set" in n for n in names)

    def test_report_draft_tool_marks_pending_review_not_approved(self, analysed_session):
        analysed_session.report_builder = _volume_report
        result = agent_mod.build_registry(analysed_session).invoke(
            "generate_report_draft", {})
        assert result.status == "ok"
        assert result.data["status"] == "pending_review"
        assert result.data["requires_human_review"] is True

    def test_report_draft_refused_without_segmentation(self, tmp_path, analysed_session):
        analysed_session.segmentation = None       # only a reference annotation left
        analysed_session.report_builder = _volume_report
        result = agent_mod.build_registry(analysed_session).invoke(
            "generate_report_draft", {})
        assert result.status == "refused"
        assert "分割" in result.summary


# ==========================================================================
# 6. Report values match real tool results
# ==========================================================================
class TestReportMatchesToolResults:
    def test_report_volume_equals_the_computed_volume(self, analysed_session):
        """The number in the report must be the number the metric code produced."""
        lm = analysed_session.segmentation.label_map
        vols = metrics.all_region_volumes(
            metrics.region_masks(lm, metrics.BRATS_2020), (1.0, 1.0, 1.0))
        ctx = report_mod.ReportContext(
            case_id="BraTS20_Training_001", dataset_source="BraTS 2020",
            data_origin_label="公开数据集真实病例", modalities=["t1", "t1ce", "t2", "flair"],
            analysis_method="MONAI SegResNet", segmentation_source="AI 模型预测",
            volumes=vols, spacing=(1.0, 1.0, 1.0), shape=(8, 8, 8))
        text = report_mod.render_template(ctx)
        assert f"{vols['WT'].volume_ml:.2f} mL" in text

    def test_consistency_check_catches_a_hallucinated_volume(self):
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s", {"WT": 10.0, "TC": 5.0, "ET": 2.0})}
        report = agent_mod.verify_answer_consistency("全肿瘤区为 99.9 mL。", results)
        assert not report.ok
        assert "99.9" in report.problems[0]

    def test_consistency_check_accepts_a_rounded_true_value(self):
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s", {"WT": 10.04, "TC": 5.0, "ET": 2.0})}
        assert agent_mod.verify_answer_consistency("全肿瘤区为 10.0 mL。", results).ok

    def test_consistency_check_catches_the_wrong_case_id(self):
        results = {"get_case_info": agent_mod.ToolResult(
            "get_case_info", "ok", "s", {"case_id": "BraTS20_Training_001"})}
        report = agent_mod.verify_answer_consistency(
            "病例 BraTS20_Training_002 的体积为 1.0 mL。", results,
            expected_case_id="BraTS20_Training_001")
        assert any("病例编号" in p for p in report.problems)

    def test_consistency_check_ignores_the_label_convention_string(self):
        """Regression: "brats_2020" (a label encoding) is not a case identifier.

        A permissive pattern flagged a correct answer that merely reported the
        case's label convention as citing the wrong case.
        """
        results = {"get_case_info": agent_mod.ToolResult(
            "get_case_info", "ok", "s", {"case_id": "BraTS20_Training_001"})}
        answer = ("病例编号：BraTS20_Training_001，参考标注 label 规范为 brats_2020，"
                  "分割结果 label 编码同为 brats_2020。")
        report = agent_mod.verify_answer_consistency(
            answer, results, expected_case_id="BraTS20_Training_001")
        assert report.ok, report.problems
        assert report.checked_case_ids == ["BraTS20_Training_001"]

    def test_consistency_check_accepts_the_voxel_volume(self):
        """Regression: 0.001 mL (voxel volume) is a real returned value.

        A correct answer that explained how the volume was derived was flagged
        because the checker only looked at the three region totals.
        """
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s",
            {"unit": "mL", "voxel_volume_ml": 0.001,
             "WT": 192.53, "TC": 47.37, "ET": 30.06})}
        answer = "体积由体素数 × 体素体积（0.001 mL）计算，WT 为 192.53 mL。"
        report = agent_mod.verify_answer_consistency(answer, results)
        assert report.ok, report.problems

    def test_consistency_check_still_catches_a_fabricated_value(self):
        """Widening the collector must not blunt the check."""
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s",
            {"unit": "mL", "voxel_volume_ml": 0.001,
             "WT": 192.53, "TC": 47.37, "ET": 30.06})}
        report = agent_mod.verify_answer_consistency("全肿瘤区约 250.0 mL。", results)
        assert not report.ok

    def test_consistency_check_ignores_unrelated_numbers(self):
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s", {"WT": 10.0, "TC": 5.0, "ET": 2.0})}
        # Voxel counts and slice indices are not volumes and must not trip it.
        assert agent_mod.verify_answer_consistency(
            "位于第 68 层，共 65536 体素，全肿瘤区 10.0 mL。", results).ok

    def test_checked_values_are_reported(self):
        results = {"calculate_volume": agent_mod.ToolResult(
            "calculate_volume", "ok", "s", {"WT": 10.0, "TC": 5.0, "ET": 2.0})}
        report = agent_mod.verify_answer_consistency("WT 10.0 mL，TC 5.0 mL。", results)
        assert report.checked_volumes == [10.0, 5.0]
        assert report.ok


# ==========================================================================
# 7. Trace records real execution state and timing
# ==========================================================================
class TestTraceRecordsExecution:
    def test_tool_summaries_are_human_readable_not_repr(self, analysed_session):
        """Regression: a missing call parenthesis put a bound method in the UI."""
        result = agent_mod.build_registry(analysed_session).invoke(
            "get_segmentation_results", {})
        assert result.status == "ok"
        assert "bound method" not in result.summary
        assert "<" not in result.summary
        assert "SegResNet" in result.summary
        assert analysed_session.segmentation.model_name in result.summary

    def test_trace_carries_status_and_duration(self, analysed_session, fake_llm):
        fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "get_case_info", {})]),
            _FakeMessage(tool_calls=[tool_call("c2", "calculate_volume", {})]),
            _FakeMessage(content="完成。"),
        ])
        turn = agent_mod.run_agent_turn("分析", analysed_session, CFG, tools_supported=True)
        assert len(turn.tool_calls) == 2
        for step in turn.tool_calls:
            assert step["status"] == "ok"
            assert isinstance(step["duration_ms"], float)
            assert step["duration_ms"] >= 0.0

    def test_turn_records_model_and_tool_time(self, analysed_session, fake_llm):
        fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "get_case_info", {})]),
            _FakeMessage(content="完成。"),
        ])
        turn = agent_mod.run_agent_turn("分析", analysed_session, CFG, tools_supported=True)
        assert turn.model == "fake-model"
        assert turn.thinking_ms >= 0.0
        assert turn.tool_time_ms >= 0.0
        assert turn.total_ms == pytest.approx(turn.thinking_ms + turn.tool_time_ms)

    def test_record_is_json_serialisable_and_has_no_secrets(self, analysed_session, fake_llm):
        fake_llm([_FakeMessage(content="完成。")])
        turn = agent_mod.run_agent_turn("分析", analysed_session, CFG, tools_supported=True)
        blob = json.dumps(turn.as_record(), ensure_ascii=False)
        assert "test-key-not-real" not in blob
        assert "http://fake" not in blob
        assert "fake-model" in blob

    def test_llm_mode_is_distinct_from_fixed_workflow(self, analysed_session, fake_llm):
        fake_llm([_FakeMessage(content="完成。")])
        llm_turn = agent_mod.run_agent_turn("分析", analysed_session, CFG, tools_supported=True)
        fixed_turn = agent_mod.run_agent_turn(
            "分析", analysed_session, agent_mod.LLMConfig(reason="no key"))
        assert llm_turn.mode == "llm_tool_calling"
        assert fixed_turn.mode == "fixed_workflow"
        assert llm_turn.mode != fixed_turn.mode

    def test_multi_round_tool_orchestration(self, analysed_session, fake_llm):
        """The model chains tools across rounds, as the demo scenario requires."""
        fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "get_case_info", {})]),
            _FakeMessage(tool_calls=[tool_call("c2", "get_segmentation_results", {})]),
            _FakeMessage(tool_calls=[tool_call("c3", "calculate_volume", {})]),
            _FakeMessage(tool_calls=[tool_call("c4", "get_review_status", {})]),
            _FakeMessage(content="已完成分析并整理体积。"),
        ])
        turn = agent_mod.run_agent_turn("分析当前病例", analysed_session, CFG,
                                        tools_supported=True)
        names = [s["tool"] for s in turn.tool_calls]
        assert names == ["get_case_info", "get_segmentation_results",
                         "calculate_volume", "get_review_status"]
        assert turn.answer == "已完成分析并整理体积。"
        assert turn.consistency.ok


# ==========================================================================
# 8. Case isolation
# ==========================================================================
class TestCaseIsolation:
    def test_tools_report_the_session_case_not_a_previous_one(self, tmp_path):
        store = storage.Store(tmp_path / "t.db")
        case_a = cases_mod.Case(case_id="BraTS20_Training_001", dataset_source="BraTS 2020",
                                modalities=("t1",), image_paths={},
                                label_convention="brats_2020")
        session = agent_mod.AgentSession(case=case_a, case_id=case_a.case_id, store=store)
        first = agent_mod.build_registry(session).invoke("get_case_info", {})
        assert first.data["case_id"] == "BraTS20_Training_001"

        # Switching cases must produce a fresh registry bound to the new case.
        case_b = cases_mod.Case(case_id="BraTS20_Training_002", dataset_source="BraTS 2020",
                                modalities=("t1",), image_paths={},
                                label_convention="brats_2020")
        session.case, session.case_id = case_b, case_b.case_id
        second = agent_mod.build_registry(session).invoke("get_case_info", {})
        assert second.data["case_id"] == "BraTS20_Training_002"

    def test_volumes_come_from_the_current_case_only(self, tmp_path):
        store = storage.Store(tmp_path / "t.db")
        case = cases_mod.Case(case_id="A", dataset_source="ds", modalities=("t1",),
                              image_paths={}, label_convention="brats_2020")
        session = agent_mod.AgentSession(case=case, case_id="A", store=store)

        lm_a = np.zeros((8, 8, 8), np.uint8); lm_a[1:3, 1:3, 1:3] = 2
        session.segmentation = segmentation.SegmentationResult(
            label_map=lm_a, region_masks=metrics.region_masks(lm_a, metrics.BRATS_2020),
            channel_probabilities=None, spacing=(1, 1, 1), affine=np.eye(4))
        vol_a = agent_mod.build_registry(session).invoke("calculate_volume", {}).data["WT"]

        lm_b = np.zeros((8, 8, 8), np.uint8); lm_b[1:5, 1:5, 1:5] = 2
        session.case_id = "B"
        session.segmentation = segmentation.SegmentationResult(
            label_map=lm_b, region_masks=metrics.region_masks(lm_b, metrics.BRATS_2020),
            channel_probabilities=None, spacing=(1, 1, 1), affine=np.eye(4))
        vol_b = agent_mod.build_registry(session).invoke("calculate_volume", {}).data["WT"]

        assert vol_a != vol_b

    def test_consistency_check_uses_the_current_case_id(self):
        results = {"get_case_info": agent_mod.ToolResult(
            "get_case_info", "ok", "s", {"case_id": "BraTS20_Training_001"})}
        assert agent_mod.verify_answer_consistency(
            "病例 BraTS20_Training_001 已完成分析。", results,
            expected_case_id="BraTS20_Training_001").ok
        # Naming a different case than the one in view is a real error.
        assert not agent_mod.verify_answer_consistency(
            "病例 BraTS20_Training_007 已完成分析。", results,
            expected_case_id="BraTS20_Training_001").ok


# ==========================================================================
# Local model support: provider choice and text-form tool calls
# ==========================================================================
class TestProviderResolution:
    def _remote(self, enabled=True):
        return agent_mod.LLMConfig(api_key="k", model="deepseek-chat", base_url="http://remote",
                                   enabled=enabled, reason="" if enabled else "no key")

    def _local(self, enabled=True):
        return agent_mod.LLMConfig(api_key="local", model="qwen3-0.6b",
                                   base_url="http://127.0.0.1:8080/v1",
                                   enabled=enabled, reason="" if enabled else "not running")

    def test_explicit_local_choice(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_LOCAL,
                                           remote=self._remote(), local=self._local())
        assert cfg.enabled and cfg.model == "qwen3-0.6b"

    def test_explicit_remote_choice(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_DEEPSEEK,
                                           remote=self._remote(), local=self._local())
        assert cfg.enabled and cfg.model == "deepseek-chat"

    def test_auto_prefers_local_when_available(self):
        """Local costs nothing and keeps case data on this machine."""
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(), local=self._local())
        assert cfg.model == "qwen3-0.6b"

    def test_auto_does_not_silently_use_remote_when_local_is_down(self):
        """Switching to the remote API sends case data off this machine.

        That is a different act from running a model locally, so it must not
        happen because a local process happened to be down.
        """
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(), local=self._local(False))
        assert not cfg.enabled
        assert "不会自动改用 DeepSeek" in cfg.reason

    def test_auto_uses_remote_only_with_explicit_consent(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(), local=self._local(False),
                                           allow_remote_fallback=True)
        assert cfg.enabled and cfg.model == "deepseek-chat"
        assert cfg.fell_back
        assert "已经你确认" in cfg.fallback_reason

    def test_local_failure_never_reaches_the_remote_api_without_consent(self):
        """The single most important guarantee of this provider setup."""
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(), local=self._local(False))
        assert not cfg.enabled
        assert cfg.api_key == ""          # no remote credentials handed out

    def test_explicit_deepseek_choice_is_itself_the_consent(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_DEEPSEEK,
                                           remote=self._remote(), local=self._local(False))
        assert cfg.enabled and not cfg.fell_back

    def test_default_provider_is_the_remote_api(self):
        """A live demo cannot wait minutes per turn."""
        assert agent_mod.DEFAULT_PROVIDER == agent_mod.PROVIDER_DEEPSEEK

    def test_provider_label_distinguishes_local_from_remote(self):
        assert "不出本机" in self._local().provider_label()
        assert "第三方" in self._remote().provider_label()

    def test_choosing_local_does_not_silently_use_remote(self):
        """An explicit local choice must fail loudly, not quietly go remote."""
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_LOCAL,
                                           remote=self._remote(), local=self._local(False))
        assert not cfg.enabled
        assert "本地模型不可用" in cfg.reason

    def test_both_unavailable_reports_both_reasons(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(False), local=self._local(False),
                                           allow_remote_fallback=True)
        assert not cfg.enabled
        assert "no key" in cfg.reason and "not running" in cfg.reason

    def test_consent_without_a_working_remote_still_refuses(self):
        cfg = agent_mod.resolve_llm_config(agent_mod.PROVIDER_AUTO,
                                           remote=self._remote(False), local=self._local(False),
                                           allow_remote_fallback=True)
        assert not cfg.enabled

    def test_unreachable_local_endpoint_is_not_enabled(self):
        cfg = agent_mod.load_local_llm_config(timeout=1.0)
        # No server is started inside the test environment.
        if not cfg.enabled:
            assert "未运行" in cfg.reason or "未提供模型" in cfg.reason

    def test_remote_configuration_survives(self):
        """The DeepSeek path must keep working after adding the local option."""
        remote = agent_mod.LLMConfig(api_key="k", model="deepseek-chat", enabled=True)
        assert agent_mod.resolve_llm_config(
            agent_mod.PROVIDER_DEEPSEEK, remote=remote, local=self._local(False)).enabled


class TestTextFormToolCalls:
    def test_parses_the_format_qwen3_actually_emits(self):
        content = ('<think>\n用户要查病例，应该调用 get_case_info。\n</think>\n\n'
                   '<tool_call>\n{"name": "get_case_info", "arguments": {}}\n</tool_call>')
        calls = agent_mod.extract_text_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_case_info"
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_parses_multiple_calls(self):
        content = ('<tool_call>{"name": "calculate_volume", "arguments": {}}</tool_call>'
                   '<tool_call>{"name": "get_review_status", "arguments": {}}</tool_call>')
        names = [c["function"]["name"] for c in agent_mod.extract_text_tool_calls(content)]
        assert names == ["calculate_volume", "get_review_status"]

    def test_parses_arguments_as_a_json_string(self):
        content = '<tool_call>{"name": "x", "arguments": "{\\"confirm\\": true}"}</tool_call>'
        call = agent_mod.extract_text_tool_calls(content)[0]
        assert json.loads(call["function"]["arguments"]) == {"confirm": True}

    def test_malformed_json_is_skipped_not_guessed(self):
        assert agent_mod.extract_text_tool_calls("<tool_call>not json</tool_call>") == []

    def test_block_without_a_name_is_skipped(self):
        assert agent_mod.extract_text_tool_calls('<tool_call>{"arguments": {}}</tool_call>') == []

    def test_plain_text_yields_nothing(self):
        assert agent_mod.extract_text_tool_calls("这是一个普通回答。") == []

    def test_empty_input(self):
        assert agent_mod.extract_text_tool_calls("") == []
        assert agent_mod.extract_text_tool_calls(None) == []

    def test_strip_thinking_removes_closed_block(self):
        out = agent_mod.strip_thinking("<think>推理中…</think>\n最终答案")
        assert out == "最终答案"
        assert "推理" not in out

    def test_strip_thinking_removes_truncated_block(self):
        """A cut-off generation can leave the block unterminated."""
        out = agent_mod.strip_thinking("答案在前面。<think>被截断的推理")
        assert "被截断" not in out

    def test_strip_thinking_leaves_clean_text_alone(self):
        assert agent_mod.strip_thinking("普通回答") == "普通回答"


class TestReadToolCall:
    def test_reads_dict_form(self):
        call = {"id": "c1", "function": {"name": "t", "arguments": '{"a": 1}'}}
        assert agent_mod._read_tool_call(call) == ("t", '{"a": 1}', "c1")

    def test_reads_sdk_object_form(self):
        call = _FakeToolCall("c2", "t2", '{"b": 2}')
        assert agent_mod._read_tool_call(call) == ("t2", '{"b": 2}', "c2")


class TestRecoveredCallExecution:
    def test_loop_executes_a_text_form_call(self, analysed_session, fake_llm):
        """End-to-end: a server that returns the call as text still runs it."""
        fake_llm([
            _FakeMessage(content='<tool_call>{"name": "calculate_volume", '
                                 '"arguments": {}}</tool_call>'),
            _FakeMessage(content="体积已获取。"),
        ])
        turn = agent_mod.run_agent_turn("体积", analysed_session, CFG, tools_supported=True)
        executed = [s["tool"] for s in turn.tool_calls]
        assert executed == ["calculate_volume"]
        assert turn.tool_calls[0]["status"] == "ok"
        assert turn.recovered_calls == 1
        assert "mL" in turn.tool_calls[0]["summary"]

    def test_recovered_calls_still_pass_schema_validation(self, analysed_session, fake_llm):
        """The text path must not become a way around argument checking."""
        fake_llm([
            _FakeMessage(content='<tool_call>{"name": "run_segmentation", '
                                 '"arguments": {"confirm": true, "shell": "rm -rf /"}}'
                                 '</tool_call>'),
            _FakeMessage(content="完成。"),
        ])
        turn = agent_mod.run_agent_turn("跑分割", analysed_session, CFG, tools_supported=True)
        assert turn.tool_calls[0]["status"] == "error"
        assert "参数校验未通过" in turn.tool_calls[0]["summary"]

    def test_recovered_run_segmentation_is_still_refused(self, analysed_session, fake_llm):
        fake_llm([
            _FakeMessage(content='<tool_call>{"name": "run_segmentation", '
                                 '"arguments": {"confirm": true}}</tool_call>'),
            _FakeMessage(content="请自行点击按钮。"),
        ])
        turn = agent_mod.run_agent_turn("跑分割", analysed_session, CFG, tools_supported=True)
        assert turn.tool_calls[0]["status"] == "refused"

    def test_unknown_tool_in_text_form_is_refused(self, analysed_session, fake_llm):
        fake_llm([
            _FakeMessage(content='<tool_call>{"name": "delete_everything", '
                                 '"arguments": {}}</tool_call>'),
            _FakeMessage(content="该工具不可用。"),
        ])
        turn = agent_mod.run_agent_turn("清理", analysed_session, CFG, tools_supported=True)
        assert turn.tool_calls[0]["status"] == "refused"

    def test_native_tool_calls_still_take_priority(self, analysed_session, fake_llm):
        """Text parsing must not interfere when the server returns real tool_calls."""
        fake_llm([
            _FakeMessage(tool_calls=[tool_call("c1", "get_case_info", {})]),
            _FakeMessage(content="完成。"),
        ])
        turn = agent_mod.run_agent_turn("信息", analysed_session, CFG, tools_supported=True)
        assert turn.recovered_calls == 0
        assert turn.tool_calls[0]["tool"] == "get_case_info"

    def test_thinking_block_is_not_shown_to_the_user(self, analysed_session, fake_llm):
        fake_llm([_FakeMessage(content="<think>内部推理</think>这是回答。")])
        turn = agent_mod.run_agent_turn("问", analysed_session, CFG, tools_supported=True)
        assert turn.answer == "这是回答。"
        assert "内部推理" not in turn.answer
