"""The agent layer: a tool registry, a real tool-calling loop, and a fallback.

Design position
---------------
The language model is **not** the thing that analyses images.  A validated
segmentation network does that.  The agent's job is the one it is actually good
at: deciding which registered tool to call, in what order, and turning the
structured results into prose a clinician can skim.

Consequences of taking that seriously:

* Every tool call goes through :class:`ToolRegistry`, which validates arguments
  and returns a typed result.  A tool the user cannot legally run (for example
  segmenting before the input checks pass) fails loudly instead of pretending.
* The mode is always reported.  ``llm_tool_calling`` means the model really
  chose and invoked tools.  ``fixed_workflow`` means no API was configured and a
  deterministic Python workflow ran instead.  There is no third, quieter mode
  where the model merely *describes* tool calls it never made.
* Nothing destructive is exposed.  There is deliberately no shell tool, no file
  writer and no delete.  Report changes require a separate explicit user action.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import imaging
import metrics

# --------------------------------------------------------------------------
# Tool results
# --------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Outcome of one tool invocation."""

    tool: str
    status: str                       # "ok" | "error" | "refused"
    summary: str                      # one-line, shown in the trace UI
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_trace(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
        }


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------
class ToolArgumentError(ValueError):
    """Raised when the model supplies arguments that do not match the schema."""


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "object": (dict,),
    "array": (list, tuple),
}


def validate_arguments(name: str, schema: dict[str, Any],
                       arguments: Any) -> dict[str, Any]:
    """Validate a tool call's arguments against its declared JSON Schema.

    The model is the least trustworthy component in this pipeline, so its output
    is treated as untrusted input: the wrong type, a missing required field or an
    undeclared property is rejected with a message the model can act on, rather
    than being splatted into a Python call.
    """
    if not isinstance(arguments, dict):
        raise ToolArgumentError(
            f"{name}: 参数必须是 JSON 对象，实际为 {type(arguments).__name__}")

    props: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolArgumentError(f"{name}: 缺少必需参数 {missing}")

    unknown = [key for key in arguments if key not in props]
    if unknown:
        raise ToolArgumentError(
            f"{name}: 不接受未声明的参数 {unknown}；可用参数为 {sorted(props)}")

    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        spec = props.get(key, {})
        expected = spec.get("type")
        if expected is None or value is None:
            cleaned[key] = value
            continue

        allowed = _JSON_TYPES.get(expected)
        if allowed is None:
            cleaned[key] = value
            continue

        # bool is a subclass of int in Python; keep them distinct so that a
        # model passing `1` where `true` is expected is caught.
        if expected in ("integer", "number") and isinstance(value, bool):
            raise ToolArgumentError(f"{name}.{key}: 期望 {expected}，收到 boolean")

        if not isinstance(value, allowed):
            raise ToolArgumentError(
                f"{name}.{key}: 期望 {expected}，收到 {type(value).__name__}")

        if "enum" in spec and value not in spec["enum"]:
            raise ToolArgumentError(
                f"{name}.{key}: 取值必须是 {spec['enum']} 之一，收到 {value!r}")
        cleaned[key] = value

    return cleaned


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult]
    requires_confirmation: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the tools the agent may call, and refuses everything else."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.calls: list[dict[str, Any]] = []

    def register(self, tool: Tool) -> None:
        if not tool.name.replace("_", "").isalnum():
            raise ValueError(f"非法工具名: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run a tool, recording the call and never raising into the UI."""
        arguments = {} if arguments is None else arguments
        started = time.perf_counter()

        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                name, "refused", f"未注册的工具: {name}",
                error=f"tool '{name}' is not registered; available: {sorted(self._tools)}")
        else:
            try:
                clean = validate_arguments(name, tool.parameters, arguments)
            except ToolArgumentError as exc:
                result = ToolResult(name, "error", "参数校验未通过", error=str(exc))
            else:
                try:
                    result = tool.func(**clean)
                except TypeError as exc:
                    result = ToolResult(name, "error", "参数错误", error=str(exc))
                except Exception as exc:  # noqa: BLE001
                    result = ToolResult(name, "error",
                                        f"执行失败: {type(exc).__name__}", error=str(exc))

        result.duration_ms = (time.perf_counter() - started) * 1000.0
        self.calls.append({"arguments": arguments, **result.to_trace()})
        return result


# --------------------------------------------------------------------------
# Session state the tools operate on
# --------------------------------------------------------------------------
@dataclass
class AgentSession:
    """Everything the tools may read.  Populated by the UI layer.

    ``report_builder`` is injected by the app: it renders a structured report
    draft from the *same* deterministic context the UI uses, so an agent-drafted
    report and a button-drafted report cannot diverge.  When it is absent the
    tool refuses rather than inventing a report.
    """

    case: Any = None                    # cases.Case
    validation: Any = None              # imaging.ValidationReport
    segmentation: Any = None            # segmentation.SegmentationResult
    reference: Any = None               # segmentation.ReferenceAnnotation
    evaluation: Any = None              # metrics.SegmentationEvaluation
    review_status: str = "unreviewed"
    report_present: bool = False
    store: Any = None
    case_id: str | None = None
    last_run: dict[str, Any] = field(default_factory=dict)
    report_builder: Callable[[], tuple[str, str]] | None = None
    report_saver: Callable[[str, str], str] | None = None

    def active_label_map(self):
        """The mask currently being displayed, with its provenance."""
        if self.segmentation is not None:
            return self.segmentation.label_map, "AI 模型预测"
        if self.reference is not None:
            return self.reference.label_map, "数据集参考标注"
        return None, "无"

    def seg_done(self) -> bool:
        """Whether a *model* prediction exists for this case.

        Deliberately not satisfied by a dataset reference annotation: a report
        drafted from expert labels describes the dataset, not the model, and the
        two must never be silently interchanged.
        """
        return self.segmentation is not None


# --------------------------------------------------------------------------
# Building the registry
# --------------------------------------------------------------------------
def build_registry(session: AgentSession) -> ToolRegistry:
    """Wire the tools to a live session.  Called once per user turn."""

    reg = ToolRegistry()

    def get_case_info() -> ToolResult:
        c = session.case
        if c is None:
            return ToolResult("get_case_info", "error", "当前未选择病例",
                              error="no case selected")
        return ToolResult(
            "get_case_info", "ok", f"病例 {c.case_id}",
            {
                "case_id": c.case_id,
                "dataset_source": c.dataset_source,
                "data_origin": c.origin_label,
                "modalities": list(c.modalities),
                "modality_labels": [imaging.MODALITY_LABELS.get(m, m.upper())
                                    for m in c.modalities],
                "has_reference_annotation": c.has_reference,
                "label_convention": c.label_convention,
                "segmentation_status": c.segmentation_status,
            },
        )

    def validate_mri() -> ToolResult:
        v = session.validation
        if v is None:
            return ToolResult("validate_mri", "error", "尚未执行输入检查",
                              error="validation not run")
        return ToolResult(
            "validate_mri", "ok" if v.ok else "error",
            "输入检查通过" if v.ok else f"输入检查未通过（{len(v.errors)} 个错误）",
            {
                "ok": v.ok,
                "modalities_present": v.modalities_present,
                "shape": list(v.shape) if v.shape else None,
                "spacing_mm": list(v.spacing) if v.spacing else None,
                "errors": [i.message for i in v.errors],
                "warnings": [i.message for i in v.warnings],
            },
        )

    def run_segmentation(confirm: bool = False) -> ToolResult:
        """Deliberately gated: running a model is a user-initiated action."""
        if not confirm:
            return ToolResult(
                "run_segmentation", "refused",
                "运行分割模型需要用户明确确认",
                error="explicit user confirmation required",
            )
        return ToolResult(
            "run_segmentation", "refused",
            "请在界面上点击「运行 AI 分割」按钮以执行模型",
            error="segmentation must be triggered from the UI button",
        )

    def get_segmentation_results() -> ToolResult:
        s = session.segmentation
        if s is None:
            if session.reference is not None:
                return ToolResult(
                    "get_segmentation_results", "ok",
                    "当前为参考标注模式（非模型预测）",
                    {
                        "source": "reference_annotation",
                        "is_model_prediction": False,
                        "volumes_ml": {r: v.volume_ml for r, v in session.reference.volumes.items()},
                    },
                )
            return ToolResult("get_segmentation_results", "error",
                              "尚无分割结果", error="no segmentation available")
        vols = s.volumes
        return ToolResult(
            "get_segmentation_results", "ok",
            f"{s.provenance()}，耗时 {s.duration_s:.1f}s",
            {
                "source": "model_prediction",
                "is_model_prediction": True,
                "model": s.model_name,
                "model_version": s.model_version,
                "architecture": s.architecture,
                "device": s.device,
                "duration_s": round(s.duration_s, 2),
                "volumes_ml": {r: v.volume_ml for r, v in vols.items()},
                "voxel_counts": {r: v.voxel_count for r, v in vols.items()},
                "warnings": s.warnings,
            },
        )

    def calculate_volume() -> ToolResult:
        """Return the deterministically computed volumes.

        The agent never computes a volume itself; it reads what
        :mod:`metrics` already produced.
        """
        source = session.segmentation or session.reference
        if source is None:
            return ToolResult("calculate_volume", "error", "尚无掩膜，无法计算体积",
                              error="no mask available")
        vols = source.volumes
        return ToolResult(
            "calculate_volume", "ok",
            "WT {:.2f} mL / TC {:.2f} mL / ET {:.2f} mL".format(
                vols["WT"].volume_ml, vols["TC"].volume_ml, vols["ET"].volume_ml
            ),
            {
                "unit": "mL",
                "computed_by": "deterministic voxel-count × voxel-volume",
                "voxel_volume_ml": vols["WT"].voxel_volume_ml,
                "WT": vols["WT"].volume_ml,
                "TC": vols["TC"].volume_ml,
                "ET": vols["ET"].volume_ml,
                "source": "model_prediction" if session.segmentation is not None else "reference_annotation",
            },
        )

    def get_review_status() -> ToolResult:
        return ToolResult(
            "get_review_status", "ok",
            f"审核状态: {session.review_status}",
            {
                "review_status": session.review_status,
                "report_present": session.report_present,
                "teaching_record": bool(
                    session.case and getattr(session.case, "teaching_record", None)
                ),
            },
        )

    def get_evaluation() -> ToolResult:
        ev = session.evaluation
        if ev is None:
            has_ref = bool(session.case and session.case.has_reference)
            return ToolResult(
                "get_model_evaluation", "error" if has_ref else "ok",
                "尚无可用的模型评测结果（需要同时存在模型预测与参考标注）"
                if has_ref else "该病例没有参考标注，无法计算 Dice",
                {"available": False, "has_reference_annotation": has_ref},
            )
        return ToolResult(
            "get_model_evaluation", "ok",
            "WT Dice {:.4f} / TC Dice {:.4f} / ET Dice {:.4f}".format(
                ev.dice["WT"].dice or 0.0, ev.dice["TC"].dice or 0.0, ev.dice["ET"].dice or 0.0
            ),
            {
                "available": True,
                "compared_against": "dataset reference annotation",
                "dice": {k: v.dice for k, v in ev.dice.items()},
                "missed_voxels": ev.missed,
                "extra_voxels": ev.extra,
            },
        )

    def generate_report_draft() -> ToolResult:
        """Render a structured report draft from the same facts the UI uses.

        Drafting is additive: it writes a *new revision* whose status is
        ``pending_review``.  It never approves a report and never overwrites an
        existing revision, so an agent-drafted report still has to pass the same
        human review as one created by clicking the button.
        """
        if session.case is None:
            return ToolResult("generate_report_draft", "error", "当前未选择病例",
                              error="no case selected")
        if not session.seg_done():
            return ToolResult(
                "generate_report_draft", "refused",
                "尚无分割结果，无法生成报告草稿；请先在「智能阅片」运行 AI 分割",
                error="segmentation has not been run for this case",
            )
        if session.report_builder is None:
            return ToolResult(
                "generate_report_draft", "refused",
                "报告模板渲染器未接入，无法生成草稿",
                error="report_builder not configured",
            )

        try:
            content, generated_by = session.report_builder()
        except Exception as exc:  # noqa: BLE001
            return ToolResult("generate_report_draft", "error",
                              f"报告生成失败: {type(exc).__name__}", error=str(exc))

        revision_id = ""
        if session.report_saver is not None:
            try:
                revision_id = session.report_saver(content, generated_by)
            except Exception as exc:  # noqa: BLE001
                return ToolResult("generate_report_draft", "error",
                                  f"报告保存失败: {type(exc).__name__}", error=str(exc))

        return ToolResult(
            "generate_report_draft", "ok",
            f"已生成待审核报告草稿（{len(content)} 字符）",
            {
                "status": "pending_review",
                "generated_by": generated_by,
                "characters": len(content),
                "revision_id": revision_id,
                "requires_human_review": True,
                "markdown": content,
            },
        )

    def get_research_feedback() -> ToolResult:
        if session.store is None:
            return ToolResult("get_research_feedback", "error", "存储未初始化",
                              error="store unavailable")
        counts = session.store.feedback_counts()
        return ToolResult(
            "get_research_feedback", "ok",
            f"共 {sum(counts.values())} 条科研反馈记录",
            {"feedback_counts": counts, "total": sum(counts.values())},
        )

    reg.register(Tool(
        "get_case_info", "获取当前病例的基本信息（编号、数据来源、序列、是否有参考标注）。",
        {"type": "object", "properties": {}, "required": []}, get_case_info))
    reg.register(Tool(
        "validate_mri", "检查当前病例的 MRI 输入是否可读、序列是否齐全、几何信息是否一致。",
        {"type": "object", "properties": {}, "required": []}, validate_mri))
    reg.register(Tool(
        "run_segmentation",
        "运行脑肿瘤分割模型。属于需要用户明确确认的操作，助手无法自行触发。",
        {"type": "object", "properties": {
            "confirm": {"type": "boolean", "description": "用户是否已明确确认运行"}},
         "required": ["confirm"]}, run_segmentation, requires_confirmation=True))
    reg.register(Tool(
        "get_segmentation_results", "获取当前病例的分割结果、模型版本、推理设备与耗时。",
        {"type": "object", "properties": {}, "required": []}, get_segmentation_results))
    reg.register(Tool(
        "calculate_volume",
        "读取由确定性代码计算好的 WT/TC/ET 体积（单位 mL）。数值不由语言模型推算。",
        {"type": "object", "properties": {}, "required": []}, calculate_volume))
    reg.register(Tool(
        "get_review_status", "查询当前病例的报告审核状态。",
        {"type": "object", "properties": {}, "required": []}, get_review_status))
    reg.register(Tool(
        "get_model_evaluation",
        "获取分割结果与数据集参考标注的 Dice 评测（WT/TC/ET）及误差体素数。",
        {"type": "object", "properties": {}, "required": []}, get_evaluation))
    reg.register(Tool(
        "generate_report_draft", "确认报告草稿状态；实际生成与保存由用户在界面确认。",
        {"type": "object", "properties": {}, "required": []}, generate_report_draft))
    reg.register(Tool(
        "get_research_feedback", "查询科研反馈记录的错误分类统计。",
        {"type": "object", "properties": {}, "required": []}, get_research_feedback))

    return reg


# --------------------------------------------------------------------------
# LLM configuration
# --------------------------------------------------------------------------
@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = False
    reason: str = ""
    #: Which provider this actually is, for labelling the transcript.
    provider: str = ""
    #: True when the request is being served by the remote API because the
    #: local model was unavailable and the user agreed to the switch.  The UI
    #: says so rather than letting the substitution pass unnoticed.
    fell_back: bool = False
    fallback_reason: str = ""

    @property
    def is_local(self) -> bool:
        """Whether inference happens on this machine (no data leaves the host)."""
        base = self.base_url or ""
        return self.provider == PROVIDER_LOCAL or "127.0.0.1" in base or "localhost" in base

    def provider_label(self) -> str:
        if not self.enabled:
            return "未启用"
        if self.is_local:
            return "本地推理 · 数据不出本机"
        return "远程 API · 数据将发送至第三方"

    def describe(self) -> str:
        if self.enabled:
            where = "本地" if self.is_local else "远程"
            return f"{self.model} @ {self.base_url or 'default endpoint'}（{where}）"
        return f"未配置（{self.reason}）"


def load_llm_config() -> LLMConfig:
    """Read the API configuration from the environment.

    Keys are never hard-coded and never read from a file inside the repository;
    only the environment (optionally populated from a git-ignored ``.env``).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    key = os.environ.get("LLM_API_KEY", "").strip()
    base = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()

    if not key:
        return LLMConfig(reason="未设置 LLM_API_KEY", provider=PROVIDER_DEEPSEEK)
    if not model:
        return LLMConfig(reason="未设置 LLM_MODEL", api_key=key, base_url=base,
                         provider=PROVIDER_DEEPSEEK)
    return LLMConfig(api_key=key, base_url=base, model=model, enabled=True,
                     provider=PROVIDER_DEEPSEEK)


# --------------------------------------------------------------------------
# Local model provider
# --------------------------------------------------------------------------
#: Provider identifiers used by the UI selector.
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_LOCAL = "local"
PROVIDER_AUTO = "auto"

PROVIDER_LABELS = {
    PROVIDER_DEEPSEEK: "DeepSeek（远程 API）",
    PROVIDER_LOCAL: "本地 Qwen3-0.6B（llama.cpp）",
    PROVIDER_AUTO: "自动（优先本地；本地不可用时需你确认才转远程）",
}

#: Provider that serves the interactive demo. DeepSeek answers in seconds, which
#: is what a live walkthrough needs; the local model stays one click away.
DEFAULT_PROVIDER = PROVIDER_DEEPSEEK

#: Loopback by default: the local server is a user process on this host and is
#: never exposed beyond it.
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen3-0.6b")


def load_local_llm_config(timeout: float = 20.0) -> LLMConfig:
    """Configuration for the local llama.cpp server.

    "Configured" here means the server actually answers, not merely that a port
    is written down -- an unreachable local endpoint must fall back rather than
    fail the turn.

    The timeout is generous because llama.cpp serves one request at a time: a
    server that is mid-generation is available but briefly unresponsive, and
    treating that as "down" would silently switch provider mid-conversation.
    """
    import urllib.error
    import urllib.request

    base = LOCAL_LLM_BASE_URL
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        served = [m.get("id") for m in payload.get("data", [])]
        if served and LOCAL_LLM_MODEL not in served:
            return LLMConfig(base_url=base, model=LOCAL_LLM_MODEL,
                             reason=f"本地服务在运行，但未提供模型 {LOCAL_LLM_MODEL}"
                                    f"（可用：{served}）")
        return LLMConfig(api_key="local-no-key", base_url=base,
                         model=LOCAL_LLM_MODEL, enabled=True,
                         provider=PROVIDER_LOCAL)
    except Exception as exc:  # noqa: BLE001
        return LLMConfig(base_url=base, model=LOCAL_LLM_MODEL, provider=PROVIDER_LOCAL,
                         reason=f"本地模型服务未运行（{type(exc).__name__}）")


def resolve_llm_config(choice: str = DEFAULT_PROVIDER,
                       remote: LLMConfig | None = None,
                       local: LLMConfig | None = None,
                       allow_remote_fallback: bool = False) -> LLMConfig:
    """Pick a provider for this request.

    Both endpoints stay configured; only the choice of which to use changes.

    **The remote API is never used as a silent fallback.**  Switching to it
    sends the current case's structured data to a third-party service, which is
    a different act from running a model on this machine, so it happens only
    when the user asked for it -- either by choosing DeepSeek explicitly, or by
    choosing ``auto`` *and* granting ``allow_remote_fallback``.  Otherwise
    ``auto`` reports that the local model is unavailable and waits.
    """
    remote = remote if remote is not None else load_llm_config()
    local = local if local is not None else load_local_llm_config()

    if choice == PROVIDER_LOCAL:
        if local.enabled:
            return local
        return LLMConfig(provider=PROVIDER_LOCAL,
                         reason=f"本地模型不可用：{local.reason}；"
                                f"请运行 python scripts/local_llm_server.py")

    if choice == PROVIDER_DEEPSEEK:
        if remote.enabled:
            return remote
        return LLMConfig(provider=PROVIDER_DEEPSEEK,
                         reason=f"远程 API 不可用：{remote.reason}")

    # auto
    if local.enabled:
        return local
    if not allow_remote_fallback:
        return LLMConfig(
            provider=PROVIDER_AUTO,
            reason=f"本地模型不可用（{local.reason}）。"
                   f"为避免在未确认的情况下把病例数据发往外部服务，"
                   f"系统不会自动改用 DeepSeek —— "
                   f"请启动本地服务，或明确选择「DeepSeek（远程 API）」")
    if remote.enabled:
        cfg = copy.copy(remote)
        cfg.fell_back = True
        cfg.fallback_reason = f"本地模型不可用（{local.reason}），已经你确认改用远程 API"
        return cfg
    return LLMConfig(provider=PROVIDER_AUTO,
                     reason=f"本地与远程模型均不可用（{local.reason}；{remote.reason}）")


# tool_choice values tried in order.  Forcing a *specific* function is the only
# deterministic way to test "does this endpoint honour the tools parameter":
# with "auto" the model is entitled to reply in prose instead of calling, which
# makes the probe flaky (observed with deepseek-chat: identical requests
# sometimes called the tool and sometimes did not).
_PROBE_TOOL_CHOICES: list[Any] = [
    {"type": "function", "function": {"name": "get_case_info"}},
    "required",
    "auto",
]


def probe_tool_calling(cfg: LLMConfig, timeout: float = 20.0) -> tuple[bool, str]:
    """Check whether the configured endpoint actually supports function calling.

    We probe rather than assume, because several OpenAI-compatible providers
    silently ignore the ``tools`` parameter.  A provider that ignores tools
    would let the model *narrate* tool use without performing it, which is the
    exact failure mode this project must avoid.
    """
    if not cfg.enabled:
        return False, cfg.describe()
    try:
        from openai import OpenAI
    except ImportError:
        return False, "未安装 openai 客户端库"

    probe_tool = [{
        "type": "function",
        "function": {
            "name": "get_case_info",
            "description": "Return current case information.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]

    try:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return False, f"无法初始化客户端: {type(exc).__name__}: {exc}"

    errors: list[str] = []
    for choice in _PROBE_TOOL_CHOICES:
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user",
                           "content": "Call the get_case_info tool now. Do not reply in prose."}],
                tools=probe_tool,
                tool_choice=choice,
                max_tokens=64,
            )
        except Exception as exc:  # noqa: BLE001
            # Some providers reject "required" / a forced function; try the next.
            errors.append(f"{choice}: {type(exc).__name__}")
            continue

        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            label = choice if isinstance(choice, str) else "forced"
            return True, f"{cfg.model} 支持函数调用（tool_choice={label} 返回 tool_calls）"
        errors.append(f"{choice}: 未返回 tool_calls")

    return False, (f"{cfg.model} 未返回 tool_calls，将使用固定工作流模式"
                   + (f"（{'; '.join(errors)}）" if errors else ""))


# --------------------------------------------------------------------------
# Text-form tool calls (local models)
# --------------------------------------------------------------------------
# Some OpenAI-compatible servers apply the model's chat template but do not
# parse the model's tool-call syntax back into structured `tool_calls`.  The
# local llama.cpp server is one of them: Qwen3 correctly emits
#     <tool_call>{"name": "get_case_info", "arguments": {}}</tool_call>
# as ordinary text.
#
# This adapter only translates that transport format.  The model still decides
# which tool to call and what arguments to pass; nothing here selects a tool,
# invents arguments, or bypasses validation -- parsed calls go through exactly
# the same schema check and registry dispatch as native tool calls.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


class _RecoveredToolCalls(list):
    """List of tool calls parsed from text.

    Only exists so the call sites can treat recovered and native tool calls
    identically; the SDK returns a plain list here.
    """

    @property
    def native(self) -> bool:
        return False


class _NativeToolCalls(list):
    @property
    def native(self) -> bool:
        return True


def _read_tool_call(call: Any) -> tuple[str, str, str]:
    """Read (name, raw_arguments, id) from either call representation.

    The OpenAI SDK yields objects with a ``.function`` attribute; calls parsed
    from text are already dicts.  Both reach the same execution path.
    """
    if isinstance(call, dict):
        fn = call.get("function", {})
        return (fn.get("name", ""), fn.get("arguments", "") or "",
                call.get("id", ""))
    return (call.function.name, call.function.arguments or "", call.id)


def strip_thinking(text: str) -> str:
    """Drop Qwen3's `` thinking...<｜end▁of▁thinking｜>`` reasoning from user-facing text."""
    if not text:
        return text
    without = _THINK_BLOCK.sub("", text)
    # A truncated generation can leave the block unterminated.
    without = _UNCLOSED_THINK.sub("", without)
    return without.strip()


def extract_text_tool_calls(content: str | None) -> list[dict[str, Any]]:
    """Parse ``<tool_call>`` blocks out of a plain-text response.

    Returns entries shaped like the OpenAI SDK's tool calls, so downstream code
    has a single representation to handle.  Malformed blocks are skipped rather
    than guessed at.
    """
    if not content:
        return []
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_TOOL_CALL_BLOCK.finditer(content)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool")
        if not isinstance(name, str) or not name:
            continue
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({
            "id": f"local_call_{index}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    return calls


# --------------------------------------------------------------------------
# The tool-calling loop
# --------------------------------------------------------------------------
MAX_TOOL_ROUNDS = 6

AGENT_SYSTEM_PROMPT = """你是 Radiology Copilot 的影像分析助手，服务于影像科医生、医学影像学生与科研人员。

你可以调用已注册的工具来获取真实数据。规则：
1. 所有事实必须来自工具返回结果。禁止凭经验或常识编造病例信息、体积、Dice 或模型性能。
2. 你无法自行运行分割模型；若用户要求分析而尚无结果，请引导其在「智能阅片」页面点击「运行 AI 分割」。
3. 不得推断肿瘤病理类型、分级或治疗建议。
4. 明确区分「AI 模型预测」与「数据集参考标注」，两者不可混为一谈。
5. 当工具返回错误或数据缺失时，如实说明，不得用推测填补。
6. 回答使用中文，简洁、专业、可核查。必要时引用工具返回的具体数值。
7. 引用病例编号与体积时必须原样使用工具返回值，不得改写或估算。
8. 不得在回答中写入 Dice、漏分割体素数等科研评测指标。"""


# --------------------------------------------------------------------------
# Answer / tool-result consistency
# --------------------------------------------------------------------------
# Volumes the model may quote are matched with their unit so unrelated integers
# (slice counts, voxel counts) are not mistaken for measurements.
_VOLUME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*mL", re.IGNORECASE)
# Case identifiers look like `BraTS20_Training_001`.  The digit groups are
# mandatory: an earlier permissive version (`BraTS\d*_[A-Za-z]*_?\d+`) also
# matched the *label convention* string "brats_2020", so a model that correctly
# reported the encoding was flagged as citing the wrong case.
_CASE_ID_PATTERN = re.compile(r"\bBraTS\d{2,4}_[A-Za-z]+_\d{1,4}\b", re.IGNORECASE)
_VOLUME_TOLERANCE = 0.05          # mL; absorbs rounding in the model's prose


@dataclass
class ConsistencyReport:
    """Differences between what the model said and what the tools returned."""

    checked_volumes: list[float] = field(default_factory=list)
    checked_case_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_volumes": self.checked_volumes,
            "checked_case_ids": self.checked_case_ids,
            "problems": self.problems,
        }


# Field names whose numeric values are volumes in millilitres.
_ML_KEYS = {"WT", "TC", "ET", "ml", "volumes_ml", "voxel_volume_ml", "volume_ml"}


def _collect_ml_values(node: Any, out: set[float], parent_key: str = "") -> None:
    """Recursively collect every millilitre-denominated number a tool returned.

    Scanned structurally rather than by listing keys, so a correct answer that
    quotes, say, the voxel volume (0.001 mL) is not flagged as a mismatch.  The
    check exists to catch fabricated numbers, not to punish a model for quoting
    a real one the checker forgot about.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in _ML_KEYS or parent_key in ("volumes_ml",) or key.endswith("_ml"):
                    out.add(round(float(value), 2))
            else:
                _collect_ml_values(value, out, key)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect_ml_values(value, out, parent_key)


def _reported_volumes(tool_results: dict[str, "ToolResult"]) -> set[float]:
    """Every volume (mL) that some successful tool actually returned."""
    reported: set[float] = set()
    for res in tool_results.values():
        if res.status != "ok":
            continue
        _collect_ml_values(res.data, reported)
    return reported


def verify_answer_consistency(answer: str, tool_results: dict[str, "ToolResult"],
                              expected_case_id: str | None = None) -> ConsistencyReport:
    """Check the model's prose against the values the tools actually returned.

    The prompt instructs the model to quote tool values verbatim, but a prompt
    is not a guarantee.  This runs afterwards: any volume the answer claims must
    match a volume some tool reported, and any case id it names must be the case
    in view.  Discrepancies are surfaced to the user rather than quietly
    trusted -- and are never used to rewrite the answer.
    """
    report = ConsistencyReport()
    if not answer:
        return report

    reported = _reported_volumes(tool_results)
    for match in _VOLUME_PATTERN.finditer(answer):
        claimed = float(match.group(1))
        report.checked_volumes.append(claimed)
        if not reported:
            continue
        if not any(abs(claimed - r) <= _VOLUME_TOLERANCE for r in reported):
            report.problems.append(
                f"回答中的体积 {claimed:g} mL 与工具返回值不匹配"
                f"（工具返回：{sorted(reported)} mL）")

    if expected_case_id:
        for match in _CASE_ID_PATTERN.finditer(answer):
            named = match.group(0)
            report.checked_case_ids.append(named)
            if named.lower() != expected_case_id.lower():
                report.problems.append(
                    f"回答中提到的病例编号 {named} 与当前病例 {expected_case_id} 不一致")

    return report


@dataclass
class AgentTurn:
    """One complete user turn, including the real tool trace."""

    user_message: str
    mode: str                                  # llm_tool_calling | fixed_workflow
    steps: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    model: str = ""
    error: str = ""
    thinking_ms: float = 0.0                   # wall time spent inside the model
    model_calls: int = 0                       # round trips to the model API
    recovered_calls: int = 0                   # tool calls parsed from text
    consistency: ConsistencyReport = field(default_factory=ConsistencyReport)
    # Full tool results, keyed by tool name.  Kept because the trace strips the
    # `data` payload, and the consistency check needs the real returned values.
    results: dict[str, "ToolResult"] = field(default_factory=dict)

    def note_recovered_call(self) -> None:
        """Record that a round returned its tool call as text, not structure."""
        self.recovered_calls += 1

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [s for s in self.steps if s.get("kind") == "tool_call"]

    @property
    def tool_time_ms(self) -> float:
        return sum(float(s.get("duration_ms") or 0.0) for s in self.tool_calls)

    @property
    def total_ms(self) -> float:
        return self.thinking_ms + self.tool_time_ms

    def trace_markdown(self) -> str:
        if not self.steps:
            return "_本次未调用任何工具。_"
        lines = []
        for i, s in enumerate(self.steps, 1):
            if s.get("kind") == "tool_call":
                icon = {"ok": "[完成]", "error": "[异常]", "refused": "[拒绝]"}.get(s["status"], "•")
                ms = s.get("duration_ms")
                timing = f" `{ms:.0f}ms`" if isinstance(ms, (int, float)) else ""
                lines.append(f"{i}. {icon} `{s['tool']}`{timing} — {s['summary']}")
                if s.get("error"):
                    lines.append(f"   - {s['error']}")
        return "\n".join(lines) if lines else "_本次未调用任何工具。_"

    def as_record(self, include_answer: bool = True) -> dict[str, Any]:
        """Anonymised execution record, safe to keep as E2E evidence.

        Contains model name, timings, tool calls and outcomes.  It never
        contains the API key, the base URL or any request header.
        """
        return {
            "user_message": self.user_message,
            "mode": self.mode,
            "model": self.model,
            "model_calls": self.model_calls,
            "recovered_calls": self.recovered_calls,
            "thinking_ms": round(self.thinking_ms, 1),
            "tool_time_ms": round(self.tool_time_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "error": self.error,
            "consistency": self.consistency.as_dict(),
            "tool_calls": [
                {
                    "tool": s["tool"],
                    "status": s["status"],
                    "summary": s["summary"],
                    "duration_ms": s.get("duration_ms"),
                    "arguments": s.get("arguments", {}),
                    "error": s.get("error", ""),
                }
                for s in self.tool_calls
            ],
            **({"answer": self.answer} if include_answer else {}),
        }


def run_agent_turn(
    message: str,
    session: AgentSession,
    cfg: LLMConfig | None = None,
    *,
    tools_supported: bool | None = None,
) -> AgentTurn:
    """Execute one agent turn.

    Uses real function calling when the endpoint supports it, and otherwise
    falls back to a deterministic workflow whose behaviour is fully visible in
    the returned trace.
    """
    cfg = cfg or load_llm_config()
    registry = build_registry(session)

    if not cfg.enabled:
        return _fixed_workflow(message, session, registry,
                               note=f"未配置语言模型 API（{cfg.reason}）")
    if tools_supported is False:
        return _fixed_workflow(message, session, registry,
                               note="当前 API 未通过函数调用能力探测")

    return _llm_tool_loop(message, session, registry, cfg)


def _llm_tool_loop(message: str, session: AgentSession, registry: ToolRegistry,
                   cfg: LLMConfig) -> AgentTurn:
    from openai import OpenAI

    turn = AgentTurn(user_message=message, mode="llm_tool_calling", model=cfg.model)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    try:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=90)
    except Exception as exc:  # noqa: BLE001
        turn.error = f"无法初始化客户端: {exc}"
        return _fixed_workflow(message, session, registry, note=turn.error)

    for _round in range(MAX_TOOL_ROUNDS):
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=registry.schemas(),
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            turn.thinking_ms += (time.perf_counter() - started) * 1000.0
            turn.error = f"模型调用失败: {type(exc).__name__}: {exc}"
            fallback = _fixed_workflow(message, session, registry, note=turn.error)
            fallback.model = cfg.model
            fallback.thinking_ms = turn.thinking_ms
            return fallback
        turn.thinking_ms += (time.perf_counter() - started) * 1000.0
        turn.model_calls += 1

        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        # Servers that do not parse tool-call syntax return the call as text.
        # Translating it here keeps a single code path below.
        if not tool_calls:
            recovered = extract_text_tool_calls(choice.content)
            if recovered:
                tool_calls = _RecoveredToolCalls(recovered)
                turn.note_recovered_call()

        if not tool_calls:
            turn.answer = strip_thinking(choice.content or "").strip()
            turn.consistency = verify_answer_consistency(
                turn.answer, turn.results, expected_case_id=session.case_id)
            return turn

        # Record the assistant's tool request, then actually execute it.
        # Built through the normaliser so both call shapes work here too; a
        # recovered call was already a dict and would otherwise raise.
        serialised = []
        for tc in tool_calls:
            tc_name, tc_args, tc_id = _read_tool_call(tc)
            serialised.append({
                "id": tc_id or f"call_{len(serialised)}",
                "type": "function",
                "function": {"name": tc_name, "arguments": tc_args},
            })
        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": serialised,
        })

        for tc in tool_calls:
            name, raw_args, call_id = _read_tool_call(tc)
            try:
                args = json.loads(raw_args or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}

            result = registry.invoke(name, args)
            turn.results[name] = result
            turn.steps.append({"kind": "tool_call", **result.to_trace(),
                               "arguments": args})

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {"status": result.status, "summary": result.summary,
                     "data": result.data, "error": result.error},
                    ensure_ascii=False, default=str,
                ),
            })

    turn.answer = "已达到单轮工具调用上限，请缩小问题范围后重试。"
    turn.consistency = verify_answer_consistency(
        turn.answer, turn.results, expected_case_id=session.case_id)
    return turn


# --------------------------------------------------------------------------
# Deterministic fallback workflow
# --------------------------------------------------------------------------
_FALLBACK_PLAN: list[tuple[str, tuple[str, ...]]] = [
    ("get_case_info", ("病例", "case", "信息", "资料", "当前")),
    ("validate_mri", ("检查", "校验", "验证", "输入", "数据质量", "validate")),
    ("get_segmentation_results", ("分割", "结果", "肿瘤", "病灶", "segment")),
    ("calculate_volume", ("体积", "容积", "多少", "mL", "ml", "volume", "测量")),
    ("get_model_evaluation", ("dice", "Dice", "评测", "评估", "指标", "误差", "准确")),
    ("get_review_status", ("审核", "状态", "报告", "review")),
    ("get_research_feedback", ("科研", "反馈", "统计", "记录")),
]


def _select_fallback_tools(message: str) -> list[str]:
    """Keyword routing for the no-API path. Deterministic and inspectable."""
    msg = message.lower()
    chosen: list[str] = []
    for tool, keywords in _FALLBACK_PLAN:
        if any(k.lower() in msg for k in keywords):
            chosen.append(tool)

    if not chosen:
        # A generic "analyse this case" request maps onto the standard chain.
        chosen = ["get_case_info", "validate_mri", "get_segmentation_results",
                  "calculate_volume", "get_review_status"]
    else:
        if "calculate_volume" not in chosen and "get_segmentation_results" in chosen:
            chosen.append("calculate_volume")
        if "get_case_info" not in chosen:
            chosen.insert(0, "get_case_info")
    # stable de-duplication, preserving plan order
    order = {name: i for i, (name, _) in enumerate(_FALLBACK_PLAN)}
    return sorted(dict.fromkeys(chosen), key=lambda n: order.get(n, 99))


def _fixed_workflow(message: str, session: AgentSession, registry: ToolRegistry,
                    note: str = "") -> AgentTurn:
    """Run a deterministic tool chain and summarise the real results."""
    turn = AgentTurn(user_message=message, mode="fixed_workflow")

    # Keep the full ToolResult for summarising, and the trimmed trace for the
    # UI.  ``to_trace`` deliberately omits the data payload, so summarising from
    # the trace alone would lose the values we need to report.
    results: dict[str, ToolResult] = {}
    for tool in _select_fallback_tools(message):
        result = registry.invoke(tool, {})
        results[tool] = result
        turn.steps.append({"kind": "tool_call", "arguments": {},
                           **result.to_trace()})

    turn.results = results
    turn.answer = _summarise_fallback(turn, results, note)
    # The summary is built from tool values here, so it should always agree --
    # running the check anyway means the guarantee is verified, not assumed.
    turn.consistency = verify_answer_consistency(
        turn.answer, results, expected_case_id=session.case_id)
    return turn


def _summarise_fallback(turn: AgentTurn, results: dict[str, ToolResult], note: str) -> str:
    if not note and not results:
        return "未能从工具获得可用数据。"

    parts: list[str] = []
    if note:
        parts.append(f"> 固定工作流模式：{note}。以下内容由确定性 Python 工具直接返回。\n")

    by_tool = results

    case_step = by_tool.get("get_case_info")
    if case_step and case_step.ok:
        d = case_step.data
        parts.append(f"**当前病例**：{d['case_id']}（{d['data_origin']}）")
        parts.append(f"- 数据来源：{d['dataset_source']}")
        parts.append(f"- 序列：{'、'.join(d['modality_labels'])}")
        parts.append(f"- 参考标注：{'有' if d['has_reference_annotation'] else '无'}")

    seg = by_tool.get("get_segmentation_results")
    if seg:
        if seg.ok and seg.data.get("is_model_prediction"):
            d = seg.data
            parts.append(f"\n**分割结果**：模型预测 — {d['model']} v{d['model_version']} "
                         f"({d['architecture']}, {d['device']}, {d['duration_s']}s)")
        elif seg.ok:
            parts.append("\n**当前显示的是数据集参考标注，不是模型预测。**")
        else:
            parts.append(f"\n**分割结果**：{seg.summary}。请先在「智能阅片」页面运行 AI 分割。")

    vol = by_tool.get("calculate_volume")
    if vol and vol.ok:
        d = vol.data
        src = "模型预测" if d["source"] == "model_prediction" else "数据集参考标注"
        parts.append(f"\n**区域体积**（来源：{src}，由确定性代码计算）")
        parts.append(f"- WT 全肿瘤区：{d['WT']:.2f} mL")
        parts.append(f"- TC 肿瘤核心区：{d['TC']:.2f} mL")
        parts.append(f"- ET 强化肿瘤区：{d['ET']:.2f} mL")

    ev = by_tool.get("get_model_evaluation")
    if ev:
        if ev.ok and ev.data.get("available"):
            dice = ev.data["dice"]
            parts.append("\n**与参考标注的 Dice**")
            for region in ("WT", "TC", "ET"):
                val = dice.get(region)
                parts.append(f"- {region}：{'未定义' if val is None else f'{val:.4f}'}")
        else:
            parts.append(f"\n**模型评测**：{ev.summary}")

    rev = by_tool.get("get_review_status")
    if rev and rev.ok:
        parts.append(f"\n**审核状态**：{rev.data['review_status']}")

    fb = by_tool.get("get_research_feedback")
    if fb and fb.ok:
        parts.append(f"\n**科研反馈**：{fb.summary}")

    if not parts:
        parts.append("未能从工具获得可用数据，请先选择病例并完成输入检查。")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Report narration (optional LLM path)
# --------------------------------------------------------------------------
@dataclass
class LlmReportOutcome:
    """Result of asking the model to write the report narrative."""

    sections: str | None = None          # None => caller must use the template
    status: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: float = 0.0
    usage_reported: bool = False

    @property
    def ok(self) -> bool:
        return self.sections is not None

    def as_record(self) -> dict[str, Any]:
        """Anonymised record. Never contains the key, URL or request headers."""
        return {
            "model": self.model,
            "ok": self.ok,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "usage": (
                {"prompt_tokens": self.prompt_tokens,
                 "completion_tokens": self.completion_tokens,
                 "total_tokens": self.total_tokens}
                if self.usage_reported else
                {"supported": False,
                 "note": "该 API 响应未返回 usage 字段，未记录 token 用量"}
            ),
        }


def generate_llm_report_sections(ctx: Any, cfg: LLMConfig | None = None) -> LlmReportOutcome:
    """Ask the model for the narrative sections only.

    An outcome with ``sections=None`` means the caller must fall back to the
    deterministic template; it is never silently replaced with fabricated prose.
    """
    import report as report_mod

    cfg = cfg or load_llm_config()
    if not cfg.enabled:
        return LlmReportOutcome(
            status=f"未配置语言模型 API（{cfg.reason}），使用确定性模板生成报告。")

    started = time.perf_counter()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=90)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=report_mod.build_llm_messages(ctx),
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        return LlmReportOutcome(
            status=f"模型调用失败（{type(exc).__name__}），已回退到确定性模板：{exc}",
            model=cfg.model, duration_ms=(time.perf_counter() - started) * 1000.0)

    duration = (time.perf_counter() - started) * 1000.0
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return LlmReportOutcome(status="模型返回为空，使用确定性模板生成报告。",
                                model=cfg.model, duration_ms=duration)

    usage = getattr(resp, "usage", None)
    return LlmReportOutcome(
        sections=report_mod.substitute_placeholders(text, ctx),
        status=f"叙述部分由 {cfg.model} 生成。",
        model=cfg.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        duration_ms=duration,
        usage_reported=usage is not None,
    )


# Terms that must never appear in a report narrative.  The prompt forbids them;
# this is the check that runs afterwards, because a prompt is not a guarantee.
_FORBIDDEN_REPORT_TERMS = (
    "胶质母细胞瘤", "星形细胞瘤", "少突胶质", "淋巴瘤", "转移瘤", "脑膜瘤",
    "WHO", "WHO分级", "II级", "III级", "IV级", "分级", "分子分型", "基因型",
    "建议手术", "建议放疗", "建议化疗", "推荐手术", "治疗方案", "预后",
    "生存期", "活检",
)


def verify_report_narrative(text: str, ctx: Any) -> ConsistencyReport:
    """Check an LLM-written narrative against the facts it was given.

    Reuses the same consistency machinery as the agent answers: volumes must
    match the computed values and the case id must be the one in view.  It
    additionally scans for clinical conclusions the system is not entitled to
    draw.
    """
    report = ConsistencyReport()
    if not text:
        return report

    reported: set[float] = set()
    for region in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET):
        value = ctx.volume_ml(region)
        if isinstance(value, (int, float)):
            reported.add(round(float(value), 2))

    for match in _VOLUME_PATTERN.finditer(text):
        claimed = float(match.group(1))
        report.checked_volumes.append(claimed)
        if reported and not any(abs(claimed - r) <= _VOLUME_TOLERANCE for r in reported):
            report.problems.append(
                f"报告叙述中的体积 {claimed:g} mL 与计算结果不符（{sorted(reported)} mL）")

    for match in _CASE_ID_PATTERN.finditer(text):
        named = match.group(0)
        report.checked_case_ids.append(named)
        if named.lower() != str(ctx.case_id).lower():
            report.problems.append(
                f"报告叙述中的病例编号 {named} 与当前病例 {ctx.case_id} 不一致")

    lowered = text
    for term in _FORBIDDEN_REPORT_TERMS:
        if term in lowered:
            report.problems.append(f"报告叙述出现不应由本系统给出的临床结论表述：{term}")

    return report
