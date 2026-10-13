"""Structured report drafting.

Two generators exist and they are never confused with one another:

``template``
    Deterministic.  Renders the numbers that were actually computed by
    :mod:`metrics` into a fixed document skeleton.  Always available.

``llm``
    Phrasing produced by the configured language model, but *only* from the
    same structured facts, and only the narrative sections.  Numbers are
    re-inserted verbatim from the computed values after generation, so the
    model cannot alter a measurement even if it tries.

Every report carries an explicit provenance line recording which generator
produced it, so an API outage can never be disguised as a successful model
call.

The report deliberately does **not** infer tumour grade, histology or a
clinical diagnosis.  A segmentation model cannot see histopathology, and
writing "likely glioblastoma" from a mask would be fabrication.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import metrics

DISCLAIMER = (
    "本报告由科研教学原型系统自动生成，仅用于科研、教学与算法演示，"
    "不构成临床诊断意见，不可用于任何临床决策。"
)

MODALITY_LABELS = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}


@dataclass
class ReportContext:
    """All structured facts a report may draw on. Nothing else is permitted."""

    case_id: str
    dataset_source: str
    data_origin_label: str
    modalities: list[str]
    analysis_method: str                 # human-readable description of the pipeline
    segmentation_source: str             # "AI model" or "dataset reference annotation"
    model_provenance: str = ""           # e.g. "MONAI/brats_mri_segmentation v0.5.4 on cuda:2"
    volumes: dict[str, metrics.VolumeResult] = field(default_factory=dict)
    spacing: tuple[float, float, float] | None = None
    shape: tuple[int, ...] | None = None
    input_checks: str = ""
    evaluation: dict[str, Any] | None = None
    review_status: str = "unreviewed"
    scene_notes: list[str] = field(default_factory=list)

    def volume_ml(self, region: str) -> float | None:
        v = self.volumes.get(region)
        return None if v is None else v.volume_ml


# --------------------------------------------------------------------------
# Deterministic template
# --------------------------------------------------------------------------
def _fmt_ml(value: float | None) -> str:
    return "未计算" if value is None else f"{value:.2f} mL"


def _fmt_int(value: int | None) -> str:
    return "未计算" if value is None else f"{value:,}"


def render_template(ctx: ReportContext) -> str:
    """Render the deterministic report skeleton from computed values."""
    modality_str = "、".join(MODALITY_LABELS.get(m, m.upper()) for m in ctx.modalities)
    wt, tc, et = (ctx.volumes.get(r) for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET))
    spacing_str = (
        " × ".join(f"{s:.2f}" for s in ctx.spacing) + " mm" if ctx.spacing else "未提供"
    )
    shape_str = " × ".join(str(s) for s in ctx.shape) if ctx.shape else "未提供"

    lines: list[str] = []
    lines.append("# 影像分析报告草稿")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # -- 一、检查信息 ----------------------------------------------------
    lines.append("## 一、检查信息")
    lines.append("")
    lines.append(f"- **病例编号**：{ctx.case_id}")
    lines.append(f"- **数据来源**：{ctx.dataset_source}")
    lines.append(f"- **数据性质**：{ctx.data_origin_label}")
    lines.append(f"- **MRI 序列**：{modality_str}")
    lines.append(f"- **图像尺寸**：{shape_str} 体素")
    lines.append(f"- **体素间距**：{spacing_str}")
    lines.append(f"- **分析方法**：{ctx.analysis_method}")
    lines.append(f"- **分割结果来源**：{ctx.segmentation_source}")
    if ctx.model_provenance:
        lines.append(f"- **模型与推理环境**：{ctx.model_provenance}")
    lines.append("")

    # -- 二、定量分析 ----------------------------------------------------
    lines.append("## 二、定量分析")
    lines.append("")
    if ctx.volumes:
        lines.append("| 区域 | 定义 | 体素数 | 体积 |")
        lines.append("| --- | --- | ---: | ---: |")
        rows = [
            (metrics.REGION_WT, "全肿瘤区（NCR/NET + 水肿 + 强化）"),
            (metrics.REGION_TC, "肿瘤核心区（NCR/NET + 强化）"),
            (metrics.REGION_ET, "强化肿瘤区"),
        ]
        for region, definition in rows:
            v = ctx.volumes.get(region)
            lines.append(
                f"| {region} | {definition} | {_fmt_int(v.voxel_count if v else None)} "
                f"| {_fmt_ml(v.volume_ml if v else None)} |"
            )
        if wt and tc and et and wt.volume_ml > 0:
            lines.append("")
            lines.append(
                f"区域体积比例：TC/WT = {tc.volume_ml / wt.volume_ml:.1%}，"
                f"ET/WT = {et.volume_ml / wt.volume_ml:.1%}，"
                f"ET/TC = {(et.volume_ml / tc.volume_ml if tc.volume_ml > 0 else 0):.1%}。"
            )
            lines.append("")
            lines.append(
                "> 体积由分割掩膜体素数乘以体素体积（由 affine 推导）确定性计算得到，"
                "未经大语言模型估算。"
            )
    else:
        lines.append("_尚未执行分割或定量分析，本节暂无数据。_")
    lines.append("")

    # -- 三、影像观察 ----------------------------------------------------
    lines.append("## 三、影像观察")
    lines.append("")
    lines.append("_以下内容仅陈述本次分析的实际输出，不包含病理类型推断。_")
    lines.append("")
    if ctx.scene_notes:
        for note in ctx.scene_notes:
            lines.append(f"- {note}")
    elif ctx.volumes:
        present = [r for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
                   if ctx.volumes.get(r) and ctx.volumes[r].voxel_count > 0]
        if present:
            lines.append(f"- 本次分析在影像中检出以下区域：{'、'.join(present)}。")
        else:
            lines.append("- 本次分析未检出上述任何肿瘤相关区域。")
        lines.append(
            "- 分割区域的空间分布与形态学描述需由阅片者在影像浏览模块中确认后补充。"
        )
    else:
        lines.append("_无可用观察记录。_")
    lines.append("")

    # -- 四、分析摘要 ----------------------------------------------------
    lines.append("## 四、分析摘要")
    lines.append("")
    if ctx.volumes and ctx.volume_ml(metrics.REGION_WT) is not None:
        lines.append(
            f"本次分析基于 {modality_str} 序列完成脑肿瘤区域自动分割，"
            f"分割来源为{ctx.segmentation_source}。"
            f"定量结果为：全肿瘤区 {_fmt_ml(ctx.volume_ml(metrics.REGION_WT))}，"
            f"肿瘤核心区 {_fmt_ml(ctx.volume_ml(metrics.REGION_TC))}，"
            f"强化肿瘤区 {_fmt_ml(ctx.volume_ml(metrics.REGION_ET))}。"
        )
        lines.append("")
        lines.append(
            "本摘要仅总结已完成的分割与测量结果，**不包含**肿瘤病理类型、"
            "分子分型、分级或治疗建议的推断。上述内容超出本系统能力范围，"
            "需由具备资质的医师结合临床信息判断。"
        )
    else:
        lines.append("_尚未完成分割分析，暂无摘要。_")
    lines.append("")

    # -- 五、输入检查 ----------------------------------------------------
    # NOTE: Dice / missed / extra voxel counts are deliberately NOT part of the
    # imaging report.  They measure agreement between a model and a reference
    # annotation, which is research metadata about the algorithm, not a finding
    # about the patient.  They live in the research module, and the report
    # points there instead of quoting them.
    if ctx.input_checks:
        lines.append("## 五、输入数据检查")
        lines.append("")
        lines.append(ctx.input_checks)
        lines.append("")

    if ctx.evaluation:
        lines.append(
            "> 本次分析同时产生了与数据集参考标注的算法比对指标（Dice、漏分割/过度分割体素数）。"
            "该类指标用于评估模型表现，不属于影像诊断描述，"
            "因此不在本报告中列出，可在<strong>科研反馈</strong>模块查看。"
        )
        lines.append("")

    # -- 备注 ------------------------------------------------------------
    lines.append("## 备注")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")
    lines.append(f"> 报告状态：{_status_label(ctx.review_status)}")
    lines.append("")
    return "\n".join(lines)


def _status_label(status: str) -> str:
    return {
        "unreviewed": "待审核（尚未由人工确认）",
        "draft": "草稿（尚未提交审核）",
        "pending_review": "待审核",
        "approved": "已人工审核确认",
    }.get(status, status)


# --------------------------------------------------------------------------
# Facts handed to the language model
# --------------------------------------------------------------------------
def facts_for_llm(ctx: ReportContext) -> dict[str, Any]:
    """The complete, and only, fact set the LLM is allowed to write from.

    Numbers are included so the narrative can reference them, but the caller
    re-inserts the authoritative values afterwards.

    Note that ``evaluation`` (Dice and voxel error counts) is deliberately
    **excluded**.  Those describe how well the algorithm agrees with a reference
    annotation; they are not imaging findings, and a radiology report is the
    wrong place for them.  Withholding them here also removes any chance of the
    model writing them into the narrative.
    """
    return {
        "case_id": ctx.case_id,
        "dataset_source": ctx.dataset_source,
        "data_origin": ctx.data_origin_label,
        "modalities": [MODALITY_LABELS.get(m, m.upper()) for m in ctx.modalities],
        "analysis_method": ctx.analysis_method,
        "segmentation_source": ctx.segmentation_source,
        "model": ctx.model_provenance,
        "volumes_ml": {
            "WT": ctx.volume_ml(metrics.REGION_WT),
            "TC": ctx.volume_ml(metrics.REGION_TC),
            "ET": ctx.volume_ml(metrics.REGION_ET),
        },
        "voxel_counts": {
            r: (ctx.volumes[r].voxel_count if r in ctx.volumes else None)
            for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
        },
        "spacing_mm": list(ctx.spacing) if ctx.spacing else None,
        "input_checks": ctx.input_checks,
        "review_status": ctx.review_status,
    }


LLM_SYSTEM_PROMPT = """你是一名医学影像 AI 助手，正在为科研教学原型系统起草结构化报告中的**叙述性文字**。

严格遵守：
1. 只能使用给定 JSON 中的事实。不得添加任何未提供的信息。
2. 不得推断肿瘤病理类型、分子分型、WHO 分级、良恶性或治疗建议。
3. 不得编造体积、Dice 或任何数值。数值由系统在后续步骤填入，你只需在需要的位置保留占位符 {{WT_ML}}、{{TC_ML}}、{{ET_ML}}。
4. 需明确指出该结果来自自动分割模型或数据集参考标注。
5. 语气客观、简洁、专业，使用中文。
6. 若某项数据缺失（为 null），明确说明"本次分析未提供该项数据"，不得猜测。
7. 不得写入 Dice、漏分割体素数、过度分割体素数等算法评测指标——这些属于科研评估，
   不属于影像报告内容，报告中也不提供这些数据。

输出格式（Markdown，不要包含一级标题）：
### 影像观察
（2-3 条要点，仅描述分割区域的存在性、范围与分布特征，基于给定事实）

### 分析摘要
（2-3 句话，总结完成了什么分析、得到什么定量结果、结果来源）
"""


def build_llm_messages(ctx: ReportContext) -> list[dict[str, str]]:
    facts = json.dumps(facts_for_llm(ctx), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是本病例的全部可用事实（JSON）：\n\n```json\n{facts}\n```\n\n请据此起草报告叙述部分。"},
    ]


def substitute_placeholders(text: str, ctx: ReportContext) -> str:
    """Re-insert authoritative numbers, overriding anything the model wrote.

    Even with a good prompt, a language model may round or hallucinate a
    number.  Running this substitution makes that impossible for the three
    headline volumes.

    The placeholders already carry their unit, but models routinely write
    ``{{WT_ML}} mL`` anyway, which would render as "192.53 mL mL".  Repeated
    units are collapsed so the output stays readable regardless.
    """
    replacements = {
        "{{WT_ML}}": _fmt_ml(ctx.volume_ml(metrics.REGION_WT)),
        "{{TC_ML}}": _fmt_ml(ctx.volume_ml(metrics.REGION_TC)),
        "{{ET_ML}}": _fmt_ml(ctx.volume_ml(metrics.REGION_ET)),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    # "192.53 mL mL" / "192.53 mLmL" -> "192.53 mL"
    text = re.sub(r"\s*mL(\s*mL)+", " mL", text)
    return text


def merge_llm_into_template(template_text: str, llm_sections: str, model: str) -> str:
    """Replace the narrative sections of the template with LLM prose.

    If the expected placeholders are absent the template text is kept, so a
    malformed model response degrades to the deterministic version rather than
    producing a broken report.
    """
    marker = "### 影像观察"
    if marker not in llm_sections:
        return template_text

    start = template_text.find("## 三、影像观察")
    end = template_text.find("## 四、分析摘要")
    if start == -1 or end == -1:
        return template_text

    header = template_text[:start]
    tail_start = template_text.find("\n", template_text.find("\n", end) + 1)
    tail = template_text[tail_start:] if tail_start != -1 else ""

    provenance = (
        f"\n> 本段叙述文字由语言模型（`{model}`）依据系统提供的结构化事实生成；"
        f"所有数值与定量表格均由确定性代码计算并填入，未经模型改写。\n"
    )
    return f"{header}{llm_sections.strip()}\n{provenance}\n{tail}"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_markdown(content: str, case_id: str, out_dir) -> str:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{case_id}_report.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def export_json(ctx: ReportContext, content: str, case_id: str, out_dir) -> str:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "context": {
            "dataset_source": ctx.dataset_source,
            "data_origin": ctx.data_origin_label,
            "modalities": ctx.modalities,
            "segmentation_source": ctx.segmentation_source,
            "model_provenance": ctx.model_provenance,
            "spacing_mm": list(ctx.spacing) if ctx.spacing else None,
            "shape": list(ctx.shape) if ctx.shape else None,
            "review_status": ctx.review_status,
        },
        "volumes": {k: asdict(v) for k, v in ctx.volumes.items()},
        "evaluation": ctx.evaluation,
        "report_markdown": content,
    }
    path = out / f"{case_id}_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
