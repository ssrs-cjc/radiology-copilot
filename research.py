"""Research feedback: turning segmentation errors into a trackable dataset.

The research module deliberately does **not** train anything.  Its job is the
one that actually precedes model improvement: recording, in a structured and
queryable form, where a released model disagrees with expert annotation, so
that a future round of work has evidence to start from.

Two accuracy rules govern every number on the dashboard:

1. Only *measured* values are aggregated.  A case with no reference annotation
   contributes nothing to Dice statistics rather than contributing a zero.
2. Empty is a legitimate state.  With no evaluated cases the dashboard says so
   instead of showing zeros that look like poor performance.
"""

from __future__ import annotations

import csv
import io
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import metrics

# Error taxonomy offered to reviewers.
ERROR_TYPES: list[tuple[str, str]] = [
    ("under_segmentation", "漏分割"),
    ("over_segmentation", "过度分割"),
    ("boundary_deviation", "边界偏差"),
    ("input_anomaly", "输入异常"),
    ("model_not_applicable", "模型不适用"),
    ("other", "其他"),
]
ERROR_TYPE_LABELS = dict(ERROR_TYPES)
ERROR_TYPE_KEYS = [k for k, _ in ERROR_TYPES]


@dataclass
class FeedbackStats:
    """Aggregate view over the recorded feedback."""

    n_evaluated_cases: int = 0
    n_feedback_records: int = 0
    mean_dice: dict[str, float] = None            # type: ignore[assignment]
    std_dice: dict[str, float] = None             # type: ignore[assignment]
    error_counts: dict[str, int] = None           # type: ignore[assignment]
    models: dict[str, int] = None                 # type: ignore[assignment]
    hard_cases: list[dict[str, Any]] = None       # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.mean_dice = self.mean_dice or {}
        self.std_dice = self.std_dice or {}
        self.error_counts = self.error_counts or {}
        self.models = self.models or {}
        self.hard_cases = self.hard_cases or []

    @property
    def has_data(self) -> bool:
        return self.n_evaluated_cases > 0 or self.n_feedback_records > 0


def _extract_dice(run: dict[str, Any]) -> dict[str, float]:
    """Pull per-region Dice out of a stored evaluation payload."""
    ev = run.get("evaluation") or {}
    if isinstance(ev, str):
        import json
        try:
            ev = json.loads(ev)
        except json.JSONDecodeError:
            ev = {}
    dice = ev.get("dice") or {}
    out: dict[str, float] = {}
    for region in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET):
        value = dice.get(region)
        if isinstance(value, (int, float)):
            out[region] = float(value)
    return out


# --------------------------------------------------------------------------
# Independent batch evaluation
# --------------------------------------------------------------------------
# `scripts/evaluate_all.py` writes a self-contained batch evaluation over every
# local case.  It is a *different kind of measurement* from the single-case runs
# registered interactively in the app: it is a deliberate, reproducible sweep
# with a fixed device and one model revision.
#
# The two must never be pooled into one mean.  Mixing "cases I happened to click
# through in the UI" with "a scripted evaluation run" silently changes what the
# number means, and the resulting n would describe neither.
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent
BATCH_EVAL_PATH = PROJECT_ROOT / "outputs" / "evaluation.json"


@dataclass
class BatchEvaluation:
    available: bool = False
    path: str = ""
    model: str = ""
    model_version: str = ""
    architecture: str = ""
    device: str = ""
    n_cases: int = 0
    mean_dice: dict[str, float] = None       # type: ignore[assignment]
    std_dice: dict[str, float] = None        # type: ignore[assignment]
    cases: list[dict[str, Any]] = None       # type: ignore[assignment]
    note: str = ""

    def __post_init__(self) -> None:
        self.mean_dice = self.mean_dice or {}
        self.std_dice = self.std_dice or {}
        self.cases = self.cases or []

    def as_dict(self) -> dict[str, Any]:
        """Shape-compatible with :func:`aggregate` so the same tiles can render."""
        return {
            "n_evaluated_cases": self.n_cases,
            "mean_dice": self.mean_dice,
            "std_dice": self.std_dice,
        }


def load_batch_evaluation(path: _Path | str | None = None) -> BatchEvaluation:
    """Read the batch evaluation produced by ``scripts/evaluate_all.py``.

    Returns an ``available=False`` record (rather than raising) when the file is
    missing, so the dashboard can simply omit the section.
    """
    import json

    p = _Path(path) if path else BATCH_EVAL_PATH
    if not p.exists():
        return BatchEvaluation(available=False, path=str(p),
                               note="尚未运行批量评测（python scripts/evaluate_all.py）")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        cases = payload.get("cases") or []
    except (json.JSONDecodeError, OSError) as exc:
        return BatchEvaluation(available=False, path=str(p),
                               note=f"批量评测文件无法解析：{type(exc).__name__}")

    per_region: dict[str, list[float]] = {r: [] for r in
                                          (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)}
    for row in cases:
        dice = (row.get("dice") or {})
        for region in per_region:
            value = dice.get(region)
            if isinstance(value, (int, float)):
                per_region[region].append(float(value))

    mean_dice = {r: mean(v) for r, v in per_region.items() if v}
    std_dice = {r: (pstdev(v) if len(v) > 1 else 0.0) for r, v in per_region.items() if v}

    return BatchEvaluation(
        available=bool(cases),
        path=str(p),
        model=payload.get("model", ""),
        model_version=payload.get("model_version", ""),
        architecture=payload.get("architecture", ""),
        device=payload.get("device", ""),
        n_cases=len(cases),
        mean_dice=mean_dice,
        std_dice=std_dice,
        cases=cases,
        note="" if cases else "批量评测文件为空",
    )


def aggregate(store, hard_dice_threshold: float = 0.60) -> FeedbackStats:
    """Build the dashboard statistics from what is actually in the database."""
    stats = FeedbackStats()

    runs = [r for r in store.evaluated_cases() if _extract_dice(r)]

    # Count *distinct cases*, not runs.  Re-segmenting the same case produces a
    # new row with an identical score; counting rows would inflate n and, worse,
    # would make an unchanged model look like a stable one (n=2, sigma=0.000)
    # when in truth only one case has been evaluated.  `evaluated_cases()`
    # returns newest-first, so the first row seen for a case is the latest.
    latest_by_case: dict[str, dict[str, Any]] = {}
    for run in runs:
        latest_by_case.setdefault(run.get("case_id"), run)
    runs = list(latest_by_case.values())

    stats.n_evaluated_cases = len(runs)

    per_region: dict[str, list[float]] = {r: [] for r in
                                          (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)}
    for run in runs:
        for region, value in _extract_dice(run).items():
            per_region[region].append(value)

    for region, values in per_region.items():
        if values:
            stats.mean_dice[region] = mean(values)
            stats.std_dice[region] = pstdev(values) if len(values) > 1 else 0.0

    stats.error_counts = store.feedback_counts()
    stats.n_feedback_records = sum(stats.error_counts.values())

    for run in runs:
        key = f"{run.get('model_name', 'unknown')} v{run.get('model_version') or '?'}"
        stats.models[key] = stats.models.get(key, 0) + 1

    # "Hard cases" are the ones where the model measurably struggled, ranked by
    # whole-tumour Dice.  A case is only listed if it actually has a score.
    ranked = []
    for run in runs:
        dice = _extract_dice(run)
        wt = dice.get(metrics.REGION_WT)
        if wt is None:
            continue
        ranked.append({
            "case_id": run.get("case_id"),
            "WT": wt,
            "TC": dice.get(metrics.REGION_TC),
            "ET": dice.get(metrics.REGION_ET),
            "mean": mean(dice.values()) if dice else None,
            "model": f"{run.get('model_name','?')} v{run.get('model_version') or '?'}",
            "device": run.get("device"),
            "created_at": run.get("created_at"),
            "duration_s": run.get("duration_s"),
        })
    ranked.sort(key=lambda r: r["WT"])
    stats.hard_cases = [r for r in ranked if r["WT"] < hard_dice_threshold]

    return stats


def dashboard_markdown(stats: FeedbackStats,
                       batch: BatchEvaluation | None = None) -> str:
    """Render the research dashboard body.

    Interactive single-case runs and the scripted batch evaluation are reported
    in two separate blocks with their own n.  See the note above
    :class:`BatchEvaluation` for why they are never pooled.
    """
    batch = batch if batch is not None else load_batch_evaluation()

    if not stats.has_data and not batch.available:
        return (
            '<div class="rc-empty">'
            "<strong>暂无科研评测数据</strong><br>"
            "本工作台只展示真实计算得到的指标，不填充占位数字。<br>"
            "在「智能阅片」对含参考标注的病例运行一次 AI 分割，"
            "或执行 <code>python scripts/evaluate_all.py</code> 做批量评测。"
            "</div>"
        )

    lines: list[str] = []

    # -- independent batch evaluation -------------------------------------
    if batch.available:
        lines.append(_h4("独立批量评测"))
        tiles = "".join(
            theme_tile(region, batch.mean_dice.get(region), batch.std_dice.get(region),
                       batch.n_cases, caption="平均 Dice")
            for region in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
        )
        lines.append('<div class="rc-metrics">' + tiles + "</div>")
        detail = " · ".join(x for x in (
            f"{batch.model} v{batch.model_version}" if batch.model else "",
            batch.architecture,
            f"设备 {batch.device}" if batch.device else "",
        ) if x)
        lines.append(
            f'<div class="rc-hint" style="margin-top:8px">'
            f"来源：<code>scripts/evaluate_all.py</code> 全量批处理，"
            f"<strong>n={batch.n_cases}</strong> 例含参考标注病例 · {detail}。"
            f"该统计与下方交互式单病例评测<strong>口径不同，未合并计算</strong>。"
            f"</div>")
        lines.append("")

    # -- interactive single-case registrations ----------------------------
    lines.append(_h4("本机交互式单病例评测"))
    if stats.n_evaluated_cases == 0:
        lines.append('<div class="rc-empty" style="padding:14px">'
                     "尚无通过界面登记的模型评测。在「智能阅片」运行一次 AI 分割即可登记。"
                     "</div>")
    else:
        tiles = "".join(
            theme_tile(region, stats.mean_dice.get(region), stats.std_dice.get(region),
                       stats.n_evaluated_cases, caption="平均 Dice")
            for region in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
        )
        lines.append('<div class="rc-metrics">' + tiles + "</div>")
        lines.append(
            f'<div class="rc-hint" style="margin-top:8px">'
            f"来源：界面逐例登记的模型评测，<strong>n={stats.n_evaluated_cases}</strong> "
            f"例<strong>不同病例</strong>（模型预测 vs 数据集专家标注，"
            f"同一病例的多次运行只计一次）。"
            + ("样本量不足 2 例，标准差不适用。" if stats.n_evaluated_cases < MIN_N_FOR_SD else "")
            + "</div>")
        lines.append("")

    # -- error taxonomy ---------------------------------------------------
    lines.append(_h4("错误分类统计"))
    if not stats.error_counts:
        lines.append('<div class="rc-empty" style="padding:14px">尚无人工标注的错误分类记录。</div>')
    else:
        total = max(sum(stats.error_counts.values()), 1)
        rows = "".join(
            f"<tr><td>{ERROR_TYPE_LABELS.get(key, key)}</td>"
            f"<td style='text-align:right'>{count}</td>"
            f"<td style='text-align:right'>{count / total:.1%}</td></tr>"
            for key, count in sorted(stats.error_counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(_table(["错误类型", "数量", "占比"], rows))

    # -- difficult cases --------------------------------------------------
    lines.append(_h4("困难病例（WT Dice &lt; 0.60）"))
    if not stats.hard_cases:
        lines.append('<div class="rc-empty" style="padding:14px">当前没有低于阈值的病例。</div>')
    else:
        fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        rows = "".join(
            f"<tr><td>{r['case_id']}</td>"
            f"<td style='text-align:right'>{fmt(r['WT'])}</td>"
            f"<td style='text-align:right'>{fmt(r['TC'])}</td>"
            f"<td style='text-align:right'>{fmt(r['ET'])}</td>"
            f"<td style='font-size:11.5px;text-align:left'>{r['model']}</td>"
            f"<td style='font-size:11.5px;text-align:left'>{r['device'] or '—'}</td></tr>"
            for r in stats.hard_cases[:15]
        )
        lines.append(_table(["病例", "WT", "TC", "ET", "模型", "设备"], rows, right_from=1))

    # -- model versions ---------------------------------------------------
    if stats.models:
        lines.append(_h4("模型版本使用情况"))
        items = "".join(
            f"<li><code>{model}</code> — {count} 例已完成评测</li>"
            for model, count in sorted(stats.models.items(), key=lambda kv: -kv[1])
        )
        lines.append(f'<ul style="font-size:12.5px;margin:6px 0 0 0;padding-left:20px">{items}</ul>')

    lines.append(
        '<div class="rc-hint" style="margin-top:10px">Dice 仅反映与特定参考标注的一致性，'
        "不代表临床诊断准确性；参考标注本身也存在观察者间差异。</div>"
    )
    return "\n".join(lines)


def _h4(text: str) -> str:
    return f'<div class="rc-card-title" style="margin-top:14px"><span class="rc-dot"></span>{text}</div>'


def _table(headers: list[str], rows_html: str, right_from: int | None = None) -> str:
    """Small HTML table. This module emits HTML rather than Markdown because the
    dashboard is rendered by a gr.HTML component, where Markdown newlines and
    pipes collapse into a single run-on paragraph."""
    right_from = len(headers) if right_from is None else right_from
    head = "".join(
        f"<th style='text-align:{'right' if i >= right_from else 'left'}'>{h}</th>"
        for i, h in enumerate(headers)
    )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px">'
        f"<thead><tr style='color:#66788A;font-size:11.5px'>{head}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


# Region colours as CSS hex, matching imaging.REGION_COLORS.
REGION_CSS_COLORS = {"WT": "#16A6A1", "TC": "#F2A33C", "ET": "#E5484D"}

# Below this many cases a standard deviation is not a meaningful statistic, and
# printing "σ = 0.000" for a single case reads as "perfectly consistent" when it
# actually means "not measurable".
MIN_N_FOR_SD = 2


def spread_label(n: int, std_value: float | None) -> str:
    """Describe dispersion honestly for the sample size that actually exists."""
    if n < MIN_N_FOR_SD:
        return f"样本量 {n}，标准差不适用"
    if std_value is None:
        return f"n={n}"
    return f"n={n} · σ={std_value:.3f}"


def theme_tile(region: str, mean_value: float | None, std_value: float | None, n: int,
               caption: str = "平均 Dice") -> str:
    colors = REGION_CSS_COLORS
    zh = {"WT": "全肿瘤区", "TC": "肿瘤核心区", "ET": "强化肿瘤区"}[region]
    if mean_value is None:
        value, unit = "—", "暂无评测"
    else:
        value, unit = f"{mean_value:.3f}", spread_label(n, std_value)
    return (
        f'<div class="rc-metric">'
        f'<div class="rc-metric-region">{region} · {zh}</div>'
        f'<div class="rc-metric-value">{value}</div>'
        f'<div class="rc-metric-unit">{caption} · {unit}</div>'
        f'<div class="rc-metric-bar" style="background:{colors[region]}"></div>'
        f"</div>"
    )


# --------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------
FEEDBACK_COLUMNS = [
    ("feedback_id", "记录ID"),
    ("case_id", "病例编号"),
    ("dataset_source", "数据集来源"),
    ("model_name", "模型名称"),
    ("model_version", "模型版本"),
    ("error_type", "错误类型"),
    ("notes", "备注"),
    ("created_at", "记录时间"),
]


def feedback_csv(records: Iterable[dict[str, Any]]) -> str:
    """Export feedback records as UTF-8 CSV (with BOM for Excel compatibility)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([label for _, label in FEEDBACK_COLUMNS])
    for rec in records:
        row = []
        for key, _ in FEEDBACK_COLUMNS:
            value = rec.get(key)
            if key == "error_type":
                value = ERROR_TYPE_LABELS.get(value, value)
            row.append("" if value is None else value)
        writer.writerow(row)
    return "﻿" + buf.getvalue()


def evaluation_csv(store) -> str:
    """Export per-case evaluation metrics as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["病例编号", "模型", "版本", "设备", "耗时(s)",
                     "WT Dice", "TC Dice", "ET Dice", "评测时间"])
    for run in store.evaluated_cases():
        dice = _extract_dice(run)
        fmt = lambda v: "" if v is None else f"{v:.4f}"  # noqa: E731
        writer.writerow([
            run.get("case_id", ""), run.get("model_name", ""), run.get("model_version", ""),
            run.get("device", ""), run.get("duration_s", ""),
            fmt(dice.get(metrics.REGION_WT)), fmt(dice.get(metrics.REGION_TC)),
            fmt(dice.get(metrics.REGION_ET)), run.get("created_at", ""),
        ])
    return "﻿" + buf.getvalue()


def write_export(text: str, out_dir: Path | str, prefix: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"{prefix}_{stamp}.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# Error visualisation
# --------------------------------------------------------------------------
# Violet = reference voxels the model missed (false negative).
# Yellow = predicted voxels with no reference support (false positive).
ERROR_LEGEND: list[tuple[str, tuple[int, int, int]]] = [
    ("漏分割 FN", (139, 92, 246)),
    ("过度分割 FP", (250, 204, 21)),
]


def build_comparison_figure(
    anatomy: np.ndarray,
    affine,
    prediction: np.ndarray,
    other: np.ndarray,
    convention: metrics.LabelConvention,
    *,
    slice_index: int | None = None,
    crop=None,
    display_size: int | None = None,
    panel: int = 260,
) -> np.ndarray:
    """Three-panel comparison: model prediction | other mask | difference.

    ``other`` is the mask the prediction is being compared against -- a dataset
    reference annotation, or a human correction.  Both are real masks that the
    caller has already validated; the UI shows an explanation instead of a
    figure when either is missing, and never substitutes placeholder data.

    The three panels share one crop box and one slice index, so the anatomy is
    pixel-identical across them and only the overlay differs -- which is what
    makes the difference panel readable.
    """
    import imaging

    n_slices = imaging.axial_slice_count(anatomy, affine)
    if slice_index is None:
        slice_index = imaging.tumour_center_slice(other, affine)
    if slice_index is None:
        slice_index = n_slices // 2
    slice_index = int(np.clip(slice_index, 0, n_slices - 1))

    # Extract the single plane first, then derive the masks from those 2-D
    # arrays.  Computing them over the full volume instead costs ~9M-voxel
    # passes four times over and made this panel take ten seconds to appear.
    anatomy_sl = imaging.take_axial_slice(anatomy, slice_index, affine)
    pred_sl = imaging.take_axial_slice(prediction, slice_index, affine)
    other_sl = imaging.take_axial_slice(other, slice_index, affine)

    pred_masks = metrics.region_masks(pred_sl, convention)
    other_masks = metrics.region_masks(other_sl, convention)
    diff = metrics.difference_maps(pred_sl, other_sl, convention)

    panels = [
        imaging.render_slice_2d(anatomy_sl, imaging.region_layers(pred_masks),
                                alpha=0.45, crop=crop, display_size=display_size),
        imaging.render_slice_2d(anatomy_sl, imaging.region_layers(other_masks),
                                alpha=0.45, crop=crop, display_size=display_size),
        imaging.render_slice_2d(anatomy_sl, imaging.error_layers(diff),
                                alpha=0.60, crop=crop, display_size=display_size),
    ]
    return _hstack_panels(panels, panel)


def _hstack_panels(panels: list[np.ndarray], panel: int, gap: int = 6) -> np.ndarray:
    """Place panels side by side, each scaled to ``panel`` px on its long side."""
    from PIL import Image

    resized = []
    for p in panels:
        img = Image.fromarray(p)
        scale = panel / float(max(img.size))
        size = (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale))))
        resized.append(np.asarray(img.resize(size, Image.BILINEAR)))

    height = max(r.shape[0] for r in resized)
    width = sum(r.shape[1] for r in resized) + gap * (len(resized) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for r in resized:
        canvas[: r.shape[0], x:x + r.shape[1]] = r
        x += r.shape[1] + gap
    return canvas


def comparison_caption(prediction_available: bool, reference_available: bool) -> str:
    """Explain precisely why the comparison figure can or cannot be drawn."""
    if prediction_available and reference_available:
        return ""
    missing = []
    if not prediction_available:
        missing.append("模型预测")
    if not reference_available:
        missing.append("参考标注")
    return (
        "无法生成误差对比图：缺少 " + "、".join(missing) + "。"
        "该图仅使用真实预测与独立参考标注计算，缺失时不会用替代数据填充。"
    )


def error_legend_html() -> str:
    items = "".join(
        f'<span class="legend-item"><span class="legend-swatch" '
        f'style="background:rgb{colour}"></span>{label}</span>'
        for label, colour in ERROR_LEGEND
    )
    return f'<div class="legend">{items}</div>'


def suggest_error_type(missed: dict[str, int], extra: dict[str, int]) -> str:
    """Pre-select an error category from the voxel-level error profile.

    Only a *suggestion* to save the reviewer time; the value written to the
    database is whatever the reviewer confirms.
    """
    total_missed = sum(missed.values()) if missed else 0
    total_extra = sum(extra.values()) if extra else 0
    if total_missed == 0 and total_extra == 0:
        return "other"
    if total_missed > 4 * max(total_extra, 1):
        return "under_segmentation"
    if total_extra > 4 * max(total_missed, 1):
        return "over_segmentation"
    return "boundary_deviation"
