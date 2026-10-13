#!/usr/bin/env python
"""Radiology Copilot - 影像智能诊教研工作台.

A research/teaching prototype that joins three things around one case object:

    diagnosis support  ->  teaching  ->  research feedback

Run it with::

    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # (not recommended on a shared host)
    python app.py --port 7861 --host 0.0.0.0

Everything the UI shows is computed from real inputs.  Where a value genuinely
does not exist (no model weights, no reference mask, no API key) the UI says so
rather than substituting a plausible number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

import agent as agent_mod  # noqa: E402
import cases as cases_mod  # noqa: E402
import corrections  # noqa: E402
import imaging  # noqa: E402
import metrics  # noqa: E402
import report as report_mod  # noqa: E402
import research  # noqa: E402
import segmentation  # noqa: E402
import storage  # noqa: E402
import teaching as teaching_mod  # noqa: E402
import theme  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
EXPORT_DIR = OUTPUT_DIR / "exports"
for d in (OUTPUT_DIR, PRED_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DISCLAIMER_FOOTER = (
    "<strong>Radiology Copilot</strong> · 科研教学原型 · "
    "仅供科研与教学演示，不用于临床诊断<br>"
    "模型：MONAI brats_mri_segmentation (SegResNet) · 数据：BraTS 2020 公开数据集 · "
    "所有指标均由确定性代码计算"
)


# ==========================================================================
# Session state
# ==========================================================================
@dataclass
class ViewerPrefs:
    modality: str = "t1ce"
    slice_index: int = 77
    show_mask: bool = True
    alpha: float = 0.45
    window_low: float = 1.0
    window_high: float = 99.0


@dataclass
class Session:
    """Everything the callbacks need to know, held in one gr.State object."""

    case_id: str | None = None
    validation: imaging.ValidationReport | None = None
    seg_done: bool = False
    seg_meta: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    review_status: str = "unreviewed"
    report_content: str = ""
    report_generated_by: str = ""
    teaching: teaching_mod.ExerciseState | None = None
    # A correction that has been validated and compared but not yet written to
    # the research record. Nothing is persisted until the user presses save.
    pending_correction: dict[str, Any] | None = None
    last_correction_revision: int = 0
    report_narrative_check: dict[str, Any] | None = None
    viewer: ViewerPrefs = field(default_factory=ViewerPrefs)
    messages: list[dict[str, str]] = field(default_factory=list)
    last_trace: list[dict[str, Any]] = field(default_factory=list)
    last_mode: str = ""
    llm_cfg: agent_mod.LLMConfig | None = None
    llm_tools_ok: bool | None = None
    #: Which provider the user picked. Both endpoints stay configured; only the
    #: choice changes. Resolved lazily and cached until the choice changes.
    #: The interactive demo defaults to the remote API because a live walkthrough
    #: cannot wait minutes per turn; the local model stays one click away.
    llm_choice: str = agent_mod.DEFAULT_PROVIDER

    def cfg(self) -> agent_mod.LLMConfig:
        if self.llm_cfg is None:
            self.llm_cfg = agent_mod.resolve_llm_config(self.llm_choice)
        return self.llm_cfg

    def set_llm_choice(self, choice: str) -> None:
        """Switch provider and drop everything cached about the previous one."""
        if choice != self.llm_choice:
            self.llm_choice = choice
            self.llm_cfg = None
            self.llm_tools_ok = None      # the new endpoint needs its own probe


# -- volume cache ----------------------------------------------------------
_VOLUME_CACHE: dict[str, imaging.Volume] = {}
_CACHE_ORDER: list[str] = []
_CACHE_LIMIT = 24


def cached_volume(path: str) -> imaging.Volume:
    """Load a NIfTI once and keep it. Slice scrolling must not re-read 18 MB."""
    key = str(path)
    if key in _VOLUME_CACHE:
        return _VOLUME_CACHE[key]
    vol = imaging.load_volume(key)
    _VOLUME_CACHE[key] = vol
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_LIMIT:
        _VOLUME_CACHE.pop(_CACHE_ORDER.pop(0), None)
    return vol


REGISTRY = cases_mod.load_registry()
STORE = storage.Store()
storage.seed_cases_from_registry(STORE, REGISTRY.cases.values())


# -- region mask cache -----------------------------------------------------
# Deriving WT/TC/ET masks means three np.isin passes over ~9M voxels. Doing
# that on every slider tick made slice scrolling take seconds, so the masks are
# computed once per (case, source) and reused.  ``invalidate_masks`` is called
# whenever a new prediction is written.
_MASK_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def cached_masks(case_id: str, kind: str, label_map: np.ndarray,
                 convention) -> dict[str, np.ndarray]:
    key = (case_id, kind)
    hit = _MASK_CACHE.get(key)
    if hit is None:
        hit = metrics.region_masks(label_map, convention)
        if len(_MASK_CACHE) > 12:          # small bound; masks are ~9 MB each
            _MASK_CACHE.clear()
        _MASK_CACHE[key] = hit
    return hit


def invalidate_masks(case_id: str) -> None:
    for key in [k for k in _MASK_CACHE if k[0] == case_id]:
        _MASK_CACHE.pop(key, None)


# -- head framing cache ----------------------------------------------------
# The crop box is derived from a whole-volume projection, so it is computed
# once per volume and reused for every slice.  That also keeps the head from
# drifting sideways while the user scrolls.
_BBOX_CACHE: dict[str, tuple[slice, slice] | None] = {}

# Long side of the rendered panel, in pixels.  Rendering above the 240-voxel
# source means the browser scales down rather than up, which keeps the anatomy
# and the overlay edges crisp.
DISPLAY_SIZE = 512


def cached_bbox(path: str, data: np.ndarray, affine: np.ndarray) -> tuple[slice, slice] | None:
    if path not in _BBOX_CACHE:
        if len(_BBOX_CACHE) > 24:
            _BBOX_CACHE.clear()
        _BBOX_CACHE[path] = imaging.head_bounding_box(data, affine)
    return _BBOX_CACHE[path]


def case_choices() -> list[tuple[str, str]]:
    """(label, value) pairs; the label makes the data provenance unmissable."""
    out = []
    for c in REGISTRY.cases.values():
        marker = "真实公开数据" if c.data_origin == cases_mod.ORIGIN_PUBLIC else "合成测试数据"
        out.append((f"{c.case_id}  ·  {marker}", c.case_id))
    return out


def get_case(case_id: str | None) -> cases_mod.Case | None:
    return REGISTRY.get(case_id) if case_id else None


def prediction_path(case_id: str) -> Path:
    return PRED_DIR / f"{case_id}_pred.nii.gz"


def load_prediction(case_id: str) -> np.ndarray | None:
    """Load a stored prediction, going through the shared volume cache.

    Reading an 83 KB NIfTI is not slow on its own, but this is called from
    several render paths per interaction, so it is cached like any other volume.
    :func:`save_prediction` evicts the entry so a fresh run is never masked by
    a stale array.
    """
    p = prediction_path(case_id)
    if not p.exists():
        return None
    return cached_volume(str(p)).data.astype(np.uint8)


def save_prediction(case_id: str, label_map: np.ndarray, affine: np.ndarray) -> Path:
    import nibabel as nib
    p = prediction_path(case_id)
    nib.save(nib.Nifti1Image(label_map.astype(np.uint8), affine), str(p))
    # Drop any cached copy so the next read reflects what is now on disk.
    _VOLUME_CACHE.pop(str(p), None)
    if str(p) in _CACHE_ORDER:
        _CACHE_ORDER.remove(str(p))
    return p


# ==========================================================================
# Shared rendering helpers
# ==========================================================================
def _empty_figure(message: str, height: int = 420) -> np.ndarray:
    """A neutral placeholder image so the panel never collapses."""
    h, w = height, int(height * 1.15)
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    canvas[::2, :, :] = 248  # faint texture, avoids a dead flat rectangle
    return canvas


def reference_answer_unlocked(session: Session) -> bool:
    """Whether reference/answer masks may be drawn for the current state.

    Server-side gate for the teaching flow.  The teaching UI hides the answer,
    but hiding it only in the front end would be trivially bypassable, so every
    render path asks this function before drawing any mask.  The decision itself
    lives in :mod:`teaching` so it can be tested without the UI.
    """
    if session.teaching is None:
        return True                     # not in a teaching exercise
    return bool(teaching_mod.reveal_allowed(session.teaching))


def render_view(session: Session, *, show_reference: bool = False) -> np.ndarray:
    """Render the current slice with whatever masks are appropriate."""
    case = get_case(session.case_id)
    if case is None:
        return _empty_figure("请先选择病例")

    prefs = session.viewer
    mod = prefs.modality if prefs.modality in case.image_paths else case.modalities[0]
    vol = cached_volume(case.image_paths[mod])

    masks: dict[str, np.ndarray] = {}
    # A teaching exercise withholds the answer until the student submits; this
    # check is what actually enforces it.
    if prefs.show_mask and reference_answer_unlocked(session):
        if show_reference or not session.seg_done:
            ref = case.reference_path()
            if ref is not None:
                masks = cached_masks(case.case_id, "reference",
                                     cached_volume(str(ref)).data.astype(np.uint8),
                                     case.convention)
        if session.seg_done:
            pred = load_prediction(case.case_id)
            if pred is not None:
                masks = cached_masks(case.case_id, "prediction", pred, case.convention)

    n_slices = imaging.axial_slice_count(vol.data, vol.affine)
    idx = int(np.clip(prefs.slice_index, 0, n_slices - 1))
    # These are percentiles of the tissue intensities, which is what the
    # sliders are labelled as.
    lo_pct = min(prefs.window_low, prefs.window_high - 1.0)
    hi_pct = max(prefs.window_high, lo_pct + 1.0)

    return imaging.render_mri_slice(
        vol.data, vol.affine, idx,
        masks=masks, show_masks=prefs.show_mask and bool(masks),
        alpha=prefs.alpha, low_pct=lo_pct, high_pct=hi_pct,
        crop=cached_bbox(case.image_paths[mod], vol.data, vol.affine),
        display_size=DISPLAY_SIZE,
    )


def current_label_map(session: Session) -> tuple[np.ndarray | None, str, str]:
    """(label_map, source_cn, kind) for whatever mask is currently authoritative.

    ``kind`` is ``"prediction"`` or ``"reference"`` and keys the mask cache, so
    callers never recompute the region masks for the same volume.
    """
    case = get_case(session.case_id)
    if case is None:
        return None, "无", "none"
    if session.seg_done:
        pred = load_prediction(case.case_id)
        if pred is not None:
            return pred, "AI 模型预测", "prediction"
    ref = case.reference_path()
    if ref is not None:
        return cached_volume(str(ref)).data.astype(np.uint8), "数据集参考标注", "reference"
    return None, "无", "none"


def measurements_html(session: Session) -> str:
    lm, source, kind = current_label_map(session)
    case = get_case(session.case_id)
    if lm is None or case is None:
        return theme.empty_state("ruler", "暂无测量数据", "请先选择病例并运行 AI 分割")

    vol = cached_volume(case.image_paths[case.modalities[0]])
    vols = metrics.all_region_volumes(cached_masks(case.case_id, kind, lm, case.convention),
                                      vol.spacing)
    colors = {"WT": theme.TEAL, "TC": "#F2A33C", "ET": "#E5484D"}
    tiles = "".join(
        theme.metric_tile(r, metrics.REGION_NAMES_ZH[r], vols[r].volume_ml, colors[r], vols[r].voxel_count)
        for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
    )
    src_pill = theme.pill(source, "ok" if "模型" in source else "info")
    return (f'<div class="rc-metrics rc-metrics--stack">{tiles}</div>'
            f'<div class="rc-hint" style="margin-top:6px">来源：{src_pill} · 由体素数与体素体积确定性计算</div>')


def evaluation_html(session: Session) -> str:
    if not session.evaluation:
        return theme.empty_state("target", "暂无评测结果", "需要「模型预测」与「参考标注」同时存在")
    ev = session.evaluation
    rows = []
    for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET):
        d = ev.get("dice", {}).get(r)
        rows.append(
            f"<tr><td><strong>{r}</strong></td>"
            f"<td style='text-align:right'>{'—' if d is None else f'{d:.4f}'}</td>"
            f"<td style='text-align:right'>{ev.get('missed_voxels',{}).get(r,0):,}</td>"
            f"<td style='text-align:right'>{ev.get('extra_voxels',{}).get(r,0):,}</td></tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px">'
        "<thead><tr style='color:#66788A;font-size:11.5px'>"
        "<th style='text-align:left'>区域</th><th style='text-align:right'>Dice</th>"
        "<th style='text-align:right'>漏分割</th><th style='text-align:right'>过度分割</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<div class="rc-hint" style="margin-top:6px">Dice 为模型预测与数据集参考标注的一致性指标，不代表临床准确性。</div>'
    )


def case_info_html(session: Session) -> str:
    """Compact case identity card.

    Only the facts a reader needs while looking at the image.  Geometry and
    encoding details are low-frequency and moved to :func:`case_meta_html`, so
    the column stays narrow and stops wrapping mid-word.
    """
    case = get_case(session.case_id)
    if case is None:
        return theme.empty_state("folder", "尚未选择病例", "请从上方下拉框选择")

    kind = "ok" if case.data_origin == cases_mod.ORIGIN_PUBLIC else "warn"
    modalities = " · ".join(imaging.MODALITY_LABELS.get(m, m.upper()) for m in case.modalities)
    ref_pill = theme.pill("有参考标注", "ok") if case.has_reference else theme.pill("无参考标注", "muted")

    return f"""
<div class="rc-casecard">
  <div class="rc-casecard-id">{case.case_id}</div>
  <div class="rc-casecard-pills">
    <span class="rc-pill rc-pill--{kind}">{case.origin_label}</span>
    {ref_pill}
  </div>
  <div class="rc-casecard-meta">
    <div><span>序列</span>{modalities}</div>
    <div><span>来源</span>{case.dataset_source}</div>
  </div>
</div>
"""


def case_meta_html(session: Session) -> str:
    """Technical metadata, kept out of the way until asked for."""
    case = get_case(session.case_id)
    if case is None:
        return "—"
    vol = cached_volume(case.image_paths[case.modalities[0]])
    pairs = [
        ("图像尺寸", " × ".join(str(s) for s in vol.data.shape[:3]) + " 体素"),
        ("体素间距", " × ".join(f"{s:.2f}" for s in vol.spacing) + " mm"),
        ("空间方向", "".join(__import__("nibabel").aff2axcodes(vol.affine))),
        ("标签编码", f"<code>{case.label_convention}</code>"),
        ("模型通道顺序", " → ".join(
            f"ch{i}:{imaging.MODALITY_LABELS.get(m, m.upper())}"
            for i, m in enumerate(case.model_channel_order))),
    ]
    return theme.kv(pairs)


# ==========================================================================
# Callbacks
# ==========================================================================
def on_case_selected(case_id: str | None, session: Session):
    session.case_id = case_id
    session.seg_done = False
    session.seg_meta = {}
    session.evaluation = {}
    session.review_status = "unreviewed"
    session.report_content = ""
    session.report_generated_by = ""
    session.report_narrative_check = None
    session.teaching = None
    # A correction prepared against the previous case must not be savable
    # against this one -- it would write case A's mask onto case B's record.
    session.pending_correction = None
    session.viewer = ViewerPrefs()

    case = get_case(case_id)
    status_html = ""
    if case is None:
        session.validation = None
        return (
            session, _empty_figure("请先选择病例"),
            gr.update(maximum=1, value=0),
            case_info_html(session), case_meta_html(session), status_html,
            theme.empty_state("ruler", "暂无测量数据", "请先选择病例"),
            theme.empty_state("target", "暂无评测结果", "请先选择病例"),
            theme.topbar(None, *_model_badge()),
        )

    rep = imaging.validate_case(case.image_paths, case.reference_path(),
                                 loader=cached_volume)
    session.validation = rep

    vol = cached_volume(case.image_paths[case.modalities[0]])
    n = imaging.axial_slice_count(vol.data, vol.affine)

    center = n // 2
    ref = case.reference_path()
    if ref is not None:
        center = imaging.tumour_center_slice(
            cached_volume(str(ref)).data.astype(np.uint8), vol.affine) or center
    session.viewer.slice_index = center

    level = "ok" if rep.ok else "err"
    status_html = theme.banner(
        ("输入检查通过 — 影像可读取，几何信息一致。" if rep.ok else "输入检查未通过，已阻止分割。")
        + (f" 提示：{len(rep.warnings)} 项。" if rep.warnings else ""),
        level,
    )

    # Restore any previously computed artefacts for this case.
    if prediction_path(case.case_id).exists():
        session.seg_done = True
        rows = STORE.list_analysis_runs(case.case_id, limit=1)
        if rows:
            session.seg_meta = rows[0]
            session.evaluation = rows[0].get("evaluation") or {}

    return (
        session, render_view(session),
        gr.update(maximum=n - 1, value=center),
        case_info_html(session), case_meta_html(session), status_html,
        measurements_html(session), evaluation_html(session),
        theme.topbar(case.case_id, *_model_badge(), origin_label=case.origin_label),
    )


def _model_badge() -> tuple[str, str]:
    """(text, css-kind) for the top bar model badge."""
    st = segmentation.model_status()
    if st["available"]:
        return (f"{st['architecture']} 就绪", "rc-badge--ok")
    return ("权重未下载", "rc-badge--warn")


def on_view_change(session: Session, modality: str, idx: int, show_mask: bool,
                   alpha: float, win_low: float, win_high: float):
    session.viewer.modality = modality or session.viewer.modality
    session.viewer.slice_index = int(idx)
    session.viewer.show_mask = bool(show_mask)
    session.viewer.alpha = float(alpha)
    session.viewer.window_low = float(win_low)
    session.viewer.window_high = float(win_high)
    mri = render_view(session)
    case = get_case(session.case_id)
    if case is None:
        return session, mri, "", ""
    vol = cached_volume(case.image_paths[case.modalities[0]])
    info = (
        f'<div class="rc-hint">切片 <strong>{int(idx) + 1}</strong> / '
        f'{imaging.axial_slice_count(vol.data, vol.affine)} · '
        f'{imaging.MODALITY_LABELS.get(session.viewer.modality, "")} · '
        f'{"掩膜显示中" if show_mask else "掩膜已隐藏"}</div>'
    )
    legend = imaging.legend_html() if (show_mask and (session.seg_done or case.has_reference)) else ""
    return session, mri, info, legend


def on_run_segmentation(session: Session):
    case = get_case(session.case_id)
    if case is None:
        yield session, _empty_figure("请先选择病例"), theme.banner("请先选择病例。", "warn"), "", "", ""
        return
    if session.validation is not None and not session.validation.ok:
        yield (session, render_view(session),
               theme.banner("输入检查未通过，已阻止运行分割模型。请先修复数据问题。", "err"), "", "", "")
        return

    if not segmentation.bundle_present():
        msg = theme.banner(
            "未找到 MONAI 预训练权重。请运行 <code>python scripts/fetch_model.py</code> 下载，"
            "或继续使用「参考标注模式」查看影像与定量结果。", "warn")
        yield session, render_view(session), msg, "", "", ""
        return

    yield (session, render_view(session),
           theme.banner("⏳ 正在加载模型权重并执行滑窗推理，请稍候…", "info"), "", "", "")

    try:
        volumes = {m: cached_volume(case.image_paths[m]).data for m in case.modalities}
        base = cached_volume(case.image_paths[case.modalities[0]])
        image_4ch = cases_mod.order_channels(case, volumes)

        t_start = __import__("time").time()
        result = segmentation.segment_volume(
            image_4ch, spacing=base.spacing, affine=base.affine)
        save_prediction(case.case_id, result.label_map, base.affine)
        invalidate_masks(case.case_id)
        session.seg_done = True

        # Deterministic measurements, straight from the label map.
        vols = result.volumes
        measurements = {r: {"ml": v.volume_ml, "voxels": v.voxel_count} for r, v in vols.items()}

        evaluation: dict[str, Any] = {}
        if case.has_reference:
            ref_lm = cached_volume(str(case.reference_path())).data.astype(np.uint8)
            ev = metrics.evaluate_segmentation(result.label_map, ref_lm, base.spacing, case.convention)
            evaluation = ev.as_dict()
            session.evaluation = evaluation

        # The post-processing rule and the nesting-conflict counts are stored
        # with the run, so a stored mask can always be traced back to how it was
        # derived and how often the model violated the nesting assumption.
        provenance_record = result.record()

        STORE.record_analysis_run(
            case.case_id, result.model_name, "success",
            model_version=result.model_version, device=result.device,
            duration_s=round(result.duration_s, 2),
            measurements={**measurements, **provenance_record},
            evaluation=evaluation,
        )
        STORE.upsert_case(
            case.case_id,
            prediction_mask_path=str(prediction_path(case.case_id).relative_to(PROJECT_ROOT)),
            model_name=result.model_name, model_version=result.model_version,
            segmentation_status="completed",
            measurements={**measurements, **provenance_record},
        )
        session.seg_meta = {"model_name": result.model_name, "model_version": result.model_version,
                            "device": result.device, "duration_s": result.duration_s,
                            "nesting_conflicts": result.nesting.as_dict()}

        warn_html = ""
        if result.nesting.total:
            # Reported, not hidden: the raw channels are kept and only the
            # encoded label map applies the precedence rule.
            warn_html = theme.banner(
                f"后处理提示：<strong>{result.nesting.total}</strong> 个体素不满足 "
                f"ET⊆TC⊆WT（ET∉TC {result.nesting.et_outside_tc}，"
                f"TC∉WT {result.nesting.tc_outside_wt}）。原始三通道预测已完整保留，"
                f"存储结果按 ET&gt;TC&gt;WT 优先级编码，记录已随本次分析存档。",
                "warn", compact=True)

        ok = theme.banner(
            f"分割完成 — {result.provenance()}，耗时 {result.duration_s:.1f}s。"
            f"体积：WT {vols['WT'].volume_ml:.2f} mL / TC {vols['TC'].volume_ml:.2f} mL / "
            f"ET {vols['ET'].volume_ml:.2f} mL", "ok")
        yield (session, render_view(session), ok + warn_html,
               measurements_html(session), evaluation_html(session),
               imaging.legend_html())

    except segmentation.SegmentationUnavailable as exc:
        STORE.record_analysis_run(case.case_id, segmentation.MODEL_NAME, "failed",
                                  error_message=str(exc))
        yield (session, render_view(session), theme.banner(f"模型不可用：{exc}", "err"), "", "", "")
    except Exception as exc:  # noqa: BLE001
        STORE.record_analysis_run(case.case_id, segmentation.MODEL_NAME, "failed",
                                  error_message=f"{type(exc).__name__}: {exc}")
        detail = traceback.format_exc(limit=3).replace("\n", "<br>")
        yield (session, render_view(session),
               theme.banner(f"分割失败：{type(exc).__name__}: {exc}", "err")
               + f'<details style="font-size:11px;color:#66788A"><summary>技术细节</summary><pre>{detail}</pre></details>',
               "", "", "")


def on_use_reference_mode(session: Session):
    case = get_case(session.case_id)
    if case is None or not case.has_reference:
        return session, render_view(session), theme.banner("该病例没有参考标注。", "warn"), "", "", ""
    session.seg_done = False
    msg = theme.banner(
        "已切换到 <strong>参考标注模式</strong>：当前显示的是数据集专家标注，"
        "<strong>不是模型预测</strong>。可正常查看影像与定量结果，但不会产生模型评测指标。", "info")
    return session, render_view(session), msg, measurements_html(session), "", imaging.legend_html()


# ==========================================================================
# Agent
# ==========================================================================
def build_agent_session(session: Session) -> agent_mod.AgentSession:
    case = get_case(session.case_id)
    ref = None
    if case is not None and case.has_reference:
        ref = segmentation.load_reference_annotation(case.reference_path(), case.convention)

    seg_result = None
    if session.seg_done and case is not None:
        lm = load_prediction(case.case_id)
        if lm is not None:
            seg_result = segmentation.SegmentationResult(
                label_map=lm,
                region_masks=cached_masks(case.case_id, "prediction", lm, case.convention),
                channel_probabilities=None,
                spacing=cached_volume(case.image_paths[case.modalities[0]]).spacing,
                affine=cached_volume(case.image_paths[case.modalities[0]]).affine,
                device=session.seg_meta.get("device", "unknown"),
                duration_s=session.seg_meta.get("duration_s", 0.0) or 0.0,
            )

    def _build_report() -> tuple[str, str]:
        """Render the report draft from the live session (deterministic template)."""
        ctx = build_report_context(session)
        return report_mod.render_template(ctx), "template"

    def _save_report(content: str, generated_by: str) -> str:
        """Persist a draft revision.

        Mirrors what the button does -- a new revision with status
        ``pending_review``.  It deliberately does **not** approve: an
        agent-authored report still requires the same human confirmation.
        """
        STORE.save_report(case.case_id, content, "pending_review", generated_by)
        STORE.upsert_case(case.case_id, review_status="pending_review",
                          report_draft=content)
        session.report_content = content
        session.report_generated_by = generated_by
        session.review_status = "pending_review"
        return "pending_review"

    return agent_mod.AgentSession(
        case=case, validation=session.validation, segmentation=seg_result, reference=ref,
        review_status=session.review_status,
        report_present=bool(session.report_content),
        store=STORE, case_id=session.case_id,
        report_builder=_build_report,
        report_saver=_save_report,
    )


def on_llm_choice_changed(session: Session, choice: str):
    """Switch the agent's provider and report what that actually resolved to."""
    session.set_llm_choice(choice or agent_mod.PROVIDER_AUTO)
    cfg = session.cfg()
    if not cfg.enabled:
        return session, theme.banner(
            f"所选模型不可用：{cfg.reason}。"
            f"本地模型需先运行 <code>python scripts/local_llm_server.py</code>；"
            f"远程 API 需在 <code>.env</code> 中配置。", "warn")
    kind = "ok" if cfg.is_local else "warn"
    note = ""
    if not cfg.is_local:
        note = ("　⚠ 远程推理会把当前病例的结构化数据（编号、序列、体积、模型版本）"
                "发送至第三方 API；不包含影像、文件路径或患者身份信息。")
    return session, theme.banner(
        f"已选择模型：<strong>{cfg.model}</strong> · {cfg.provider_label()}。"
        f"下次提问生效，工具注册、参数校验与人工确认机制不变。{note}", kind)


def on_agent_message(message: str, chat: list, session: Session):
    """Run one agent turn, reporting progress before the blocking call.

    This is a generator so the user sees *which* model is thinking and roughly
    how long it will take.  A local CPU run takes 100-400 s; without an
    immediate status line the page simply looks frozen.
    """
    chat = chat or []
    if not message or not message.strip():
        yield chat, session, theme.trace_html([]), ""
        return

    cfg = session.cfg()

    # Say what is about to happen, before the wait.
    if not cfg.enabled:
        yield (chat, session, theme.trace_html([]),
               theme.banner(f"所选模型不可用：{cfg.reason}", "warn"))
        return

    pending = chat + [{"role": "user", "content": message}]
    if cfg.is_local:
        wait_hint = ("本地 CPU 推理，通常需要 1–7 分钟，请勿关闭页面")
    else:
        wait_hint = "远程 API，通常数秒内返回"
    yield (pending, session, theme.trace_html([]),
           theme.banner(
               f"⏳ 正在使用 <strong>{cfg.model}</strong> 推理…（{wait_hint}）", "info"))

    if session.llm_tools_ok is None and cfg.enabled:
        ok, _reason = agent_mod.probe_tool_calling(cfg)
        session.llm_tools_ok = ok

    turn = agent_mod.run_agent_turn(message, build_agent_session(session), cfg,
                                    tools_supported=session.llm_tools_ok)

    # If the model's prose disagrees with the tool values, say so next to the
    # answer. The answer is never rewritten -- the mismatch is disclosed.
    answer = turn.answer or "（无回复）"
    if turn.consistency.problems:
        bullets = "".join(f"<li>{p}</li>" for p in turn.consistency.problems)
        answer += ("\n\n> ⚠ **一致性校验提示**：以下数值与工具返回结果不符，"
                   f"请以工具结果为准。<ul>{bullets}</ul>")

    chat = chat + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    session.messages = chat
    session.last_trace = turn.steps
    session.last_mode = turn.mode

    mode_label = {
        "llm_tool_calling": "LLM 工具调用模式（真实函数调用）",
        "fixed_workflow": "固定工作流模式（确定性 Python 工具链，未经 LLM 决策）",
    }.get(turn.mode, turn.mode)
    def _fmt_ms(ms: float) -> str:
        return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"

    calls = f" · 模型调用 {turn.model_calls} 次" if turn.model_calls else ""
    timing = (f"{calls} · 模型 {_fmt_ms(turn.thinking_ms)} / "
              f"工具 {_fmt_ms(turn.tool_time_ms)}")
    checks = len(turn.consistency.checked_volumes) + len(turn.consistency.checked_case_ids)
    check_note = (f" · 一致性校验通过（{checks} 项）" if checks and turn.consistency.ok
                  else f" · ⚠ 一致性校验发现 {len(turn.consistency.problems)} 处不符"
                  if turn.consistency.problems else "")
    # Name the provider that actually served this turn. If the remote API was
    # used because the local model was down, that substitution is stated rather
    # than left for the user to infer from the model name.
    served_by = f" · 服务商 {turn.model or cfg.model}（{cfg.provider_label()}）"
    if cfg.fell_back:
        served_by += f" · ⚠ 已回退：{cfg.fallback_reason}"
    header = theme.banner(
        f"{mode_label} · {len(turn.tool_calls)} 次工具调用{timing}{served_by}{check_note}"
        + (f" · {turn.error}" if turn.error else ""),
        "warn" if (turn.error or turn.consistency.problems or cfg.fell_back) else
        ("ok" if turn.mode == "llm_tool_calling" else "info"),
    )

    STORE.record_agent_trace(session.case_id, message, turn.mode,
                             [c for c in turn.steps if c.get("kind") == "tool_call"],
                             answer, turn.model or None)

    yield chat, session, theme.trace_html(turn.steps), header


# ==========================================================================
# Report
# ==========================================================================
def build_report_context(session: Session) -> report_mod.ReportContext:
    case = get_case(session.case_id)
    if case is None:
        raise ValueError("未选择病例")
    lm, source, kind = current_label_map(session)
    base = cached_volume(case.image_paths[case.modalities[0]])
    volumes = {}
    if lm is not None:
        volumes = metrics.all_region_volumes(
            cached_masks(case.case_id, kind, lm, case.convention), base.spacing)

    method = (
        "MONAI brats_mri_segmentation (SegResNet) 滑窗推理，"
        "逐通道非零区域标准化，sigmoid 阈值 0.5，区域按 ET>TC>WT 编码"
        if session.seg_done else
        "数据集参考标注直接解析（未运行模型推理）"
    )
    provenance = ""
    if session.seg_done and session.seg_meta:
        provenance = (f"{session.seg_meta.get('model_name','?')} "
                      f"v{session.seg_meta.get('model_version','?')} "
                      f"on {session.seg_meta.get('device','?')}")

    return report_mod.ReportContext(
        case_id=case.case_id,
        dataset_source=case.dataset_source,
        data_origin_label=case.origin_label,
        modalities=list(case.modalities),
        analysis_method=method,
        segmentation_source=source,
        model_provenance=provenance,
        volumes=volumes,
        spacing=base.spacing,
        shape=tuple(base.data.shape[:3]),
        input_checks=session.validation.to_markdown() if session.validation else "",
        evaluation=session.evaluation or None,
        review_status=session.review_status,
    )


def on_generate_report(session: Session, use_llm: bool):
    case = get_case(session.case_id)
    if case is None:
        return (session, "", "_请先选择病例。_",
                theme.banner("请先选择病例。", "warn"), "")
    if not session.seg_done and not case.has_reference:
        return (session, "", "_尚无分割结果或参考标注，无法生成报告。_",
                theme.banner(
                    "尚无分割结果或参考标注，无法生成报告。请先运行 AI 分割。", "warn"), "")

    ctx = build_report_context(session)
    content = report_mod.render_template(ctx)
    generated_by = "template"
    notes = [theme.banner("已由确定性模板生成报告草稿（数值全部来自代码计算）。", "ok")]

    if use_llm:
        outcome = agent_mod.generate_llm_report_sections(ctx, session.cfg())
        if outcome.ok:
            # Verify the narrative against the same computed facts before it is
            # shown; a prompt is not a guarantee, and a wrong number in a report
            # is worse than no narrative at all.
            check = agent_mod.verify_report_narrative(outcome.sections, ctx)
            content = report_mod.merge_llm_into_template(
                content, outcome.sections, outcome.model)
            generated_by = f"llm:{outcome.model}"
            usage = (f"（token 用量 {outcome.total_tokens}）"
                     if outcome.usage_reported else "（该 API 未返回 token 用量）")
            if check.ok:
                notes = [theme.banner(
                    f"{outcome.status} 数值仍由确定性代码填入{usage}；"
                    f"一致性校验通过（核对 {len(check.checked_volumes)} 个数值）。", "ok")]
            else:
                notes = [theme.banner(
                    f"{outcome.status} 数值仍由确定性代码填入{usage}；"
                    "⚠ 一致性校验发现问题："
                    + "；".join(check.problems), "warn")]
            session.report_narrative_check = check.as_dict()
        else:
            notes = [theme.banner(f"ℹ️ {outcome.status}", "warn")]

    session.report_content = content
    session.report_generated_by = generated_by
    session.review_status = "pending_review"
    STORE.save_report(case.case_id, content, "pending_review", generated_by)
    STORE.upsert_case(case.case_id, review_status="pending_review", report_draft=content)

    return (session, content, content,
            notes[0] + theme.banner(
                "报告状态：<strong>待审核</strong>。请核对内容后点击「确认审核」。", "info"),
            generated_by)


def on_approve_report(session: Session, content: str):
    case = get_case(session.case_id)
    if case is None:
        return (session, theme.banner("请先选择病例。", "warn"), "", "")
    if not (content or "").strip():
        return (session, theme.banner("报告内容为空，无法审核。", "warn"), "", "")
    STORE.save_report(case.case_id, content, "approved", "human", approve=True)
    STORE.upsert_case(case.case_id, review_status="approved", report_draft=content)
    session.review_status = "approved"
    session.report_content = content
    return (session,
            theme.banner(
                "已确认审核。该版本已作为独立修订记录保存，原始模型预测未被覆盖。", "ok"),
            content, content)


def on_export_report(session: Session, content: str):
    case = get_case(session.case_id)
    if case is None or not (content or "").strip():
        return None, theme.banner("无可导出的报告内容。", "warn")
    ctx = build_report_context(session)
    md_path = report_mod.export_markdown(content, case.case_id, EXPORT_DIR)
    json_path = report_mod.export_json(ctx, content, case.case_id, EXPORT_DIR)
    return md_path, theme.banner(
        f"已导出：<code>{Path(md_path).name}</code> 与 <code>{Path(json_path).name}</code>"
        f"（位于 outputs/exports/）", "ok")


# ==========================================================================
# Teaching
# ==========================================================================
def on_teaching_load(session: Session, case_id: str | None):
    cid = case_id or session.case_id
    case = get_case(cid)
    if case is None:
        return (session, _empty_figure("请选择教学病例"),
                theme.banner("请选择教学病例。", "warn"), "", "", "", None)

    ref_lm = None
    if case.has_reference:
        ref_lm = cached_volume(str(case.reference_path())).data.astype(np.uint8)

    session.teaching = teaching_mod.ExerciseState(
        case_id=case.case_id,
        reference_label_map=ref_lm,
        reference_convention=case.convention,
        reference_spacing=cached_volume(case.image_paths[case.modalities[0]]).spacing,
    )

    # Teaching always shows a case's image *without* any answer overlay.
    prefs = ViewerPrefs(modality="flair", show_mask=False)
    session.viewer = prefs
    vol = cached_volume(case.image_paths[case.modalities[0]])
    n = imaging.axial_slice_count(vol.data, vol.affine)
    center = n // 2
    if ref_lm is not None:
        center = imaging.tumour_center_slice(ref_lm, vol.affine) or center
    session.viewer.slice_index = center

    task = (
        f'<div class="rc-card-title"><span class="rc-dot"></span>教学任务</div>'
        f'<p style="font-size:13px;line-height:1.7;margin:0 0 8px 0">'
        f"病例 <strong>{case.case_id}</strong>，共 {len(case.modalities)} 个序列。"
        f"请在隐藏 AI 结果的情况下独立阅片：</p>"
        f'<ol style="font-size:12.5px;line-height:1.8;margin:0;padding-left:18px;color:#243247">'
        f"<li>切换序列与层面，观察异常信号区域</li>"
        f"<li>在下方文本框记录可疑区域与影像表现</li>"
        f"<li>（可选）上传你自己制作的分割掩膜文件</li>"
        f"<li>点击「提交答案」查看参考结果与反馈</li>"
        f"</ol>"
    )
    return session, render_view(session), task, "", "", "", None


def on_teaching_submit(session: Session, answer_text: str, mask_file: str | None):
    if session.teaching is None:
        return (session,
                "**请先加载教学病例。**",
                theme.banner("请先加载教学病例。", "warn"))

    case = get_case(session.teaching.case_id)
    st = session.teaching
    submitted_lm = None
    load_note = ""

    if mask_file:
        try:
            import nibabel as nib
            img = nib.load(str(mask_file))
            arr = np.asarray(img.dataobj)
            if arr.ndim == 4:
                arr = arr[..., 0]
            submitted_lm = arr.astype(np.uint8)
            load_note = f"已读取上传掩膜，尺寸 {tuple(submitted_lm.shape)}。"
        except Exception as exc:  # noqa: BLE001
            return (session,
                    f"**掩膜文件读取失败：** {type(exc).__name__}: {exc}",
                    theme.banner(f"掩膜文件读取失败：{type(exc).__name__}: {exc}", "err"))

    evaluation: dict[str, Any] = {}
    if submitted_lm is not None and st.reference_label_map is not None:
        evaluation = teaching_mod.score_mask_submission(
            submitted_lm, st.reference_label_map, st.reference_spacing, st.reference_convention)
    elif submitted_lm is not None:
        evaluation = {"comparable": False, "reason": "该病例没有参考标注，无法计算 Dice。"}

    text_review = teaching_mod.review_text_answer(
        answer_text or "",
        {"modalities": list(case.modalities) if case else []},
        evaluation,
    )

    feedback = teaching_mod.feedback_markdown(
        evaluation, text_review, reference_available=bool(case and case.has_reference))
    if load_note:
        feedback = theme.banner(load_note, "info") + feedback

    # First submission unlocks; a repeat submission revises the record and is
    # counted, so the exercise history stays honest.
    attempt = st.record_submission(answer_text or "", mask_file, evaluation, feedback)
    repeat_note = ""
    if attempt > 1:
        repeat_note = f"（第 {attempt} 次提交，已更新记录并保留历史）"

    STORE.record_teaching_session(
        st.case_id, st.answer_text, mask_file,
        {"mask": evaluation, "text": {k: v for k, v in text_review.items() if k != "checklist"}},
        feedback,
    )

    # Now that the answer is submitted, the overlay may be revealed.
    session.viewer.show_mask = True
    status = theme.banner(
        f"已提交答案，参考结果已解锁{repeat_note}。"
        f"切换到「反馈结果」标签查看详细反馈。", "ok")
    return session, feedback, status


def on_teaching_reveal(session: Session):
    """Reveal the reference overlay — but only after the student has submitted.

    The button exists so a student can re-show the answer while reviewing their
    feedback.  It must not be usable to skip the exercise, so the submission
    state is checked here, on the server, rather than relying on the UI to hide
    the button.
    """
    decision = teaching_mod.reveal_allowed(session.teaching)
    if not decision.allowed:
        # Refuse, and make sure nothing was revealed as a side effect.
        session.viewer.show_mask = False
        return session, render_view(session), theme.banner(decision.message, "warn")

    session.viewer.show_mask = True
    return session, render_view(session), theme.banner(
        "已显示参考标注叠加。绿色=WT，橙色=TC，红色=ET。", "info")


# ==========================================================================
# Research
# ==========================================================================
def on_save_feedback(session: Session, error_type: str, notes: str):
    case = get_case(session.case_id)
    if case is None:
        return session, theme.banner("请先选择病例。", "warn"), "", research_dashboard_html()
    if not error_type:
        return session, theme.banner("请选择错误类型。", "warn"), "", research_dashboard_html()

    STORE.record_feedback(
        case.case_id, error_type,
        dataset_source=case.dataset_source,
        model_name=session.seg_meta.get("model_name") or (
            segmentation.MODEL_NAME if session.seg_done else None),
        model_version=session.seg_meta.get("model_version"),
        notes=notes or None,
        metrics=session.evaluation or {},
    )
    return (session,
            theme.banner("科研反馈已保存到本地数据库。", "ok"),
            feedback_table_html(), research_dashboard_html())


def research_dashboard_html() -> str:
    return research.dashboard_markdown(research.aggregate(STORE),
                                       research.load_batch_evaluation())


# ==========================================================================
# Human-corrected masks (research loop)
# ==========================================================================
CORRECTION_DIR_ROOT = PROJECT_ROOT


def correction_history_html(case_id: str | None) -> str:
    """Table of every correction revision stored for a case."""
    if not case_id:
        return theme.empty_state("folder", "请先选择病例", "修正历史按病例归档")
    rows_data = STORE.list_corrections(case_id)
    if not rows_data:
        return theme.empty_state(
            "upload", "尚无人工修正记录",
            "可在上方导入由 3D Slicer 等工具修改后的 NIfTI 掩膜")

    rows = []
    for r in rows_data:
        payload = r.get("metrics") or {}
        dice = payload.get("dice") or {}
        dice_txt = " / ".join(
            f"{k} {'—' if dice.get(k) is None else f'{dice[k]:.3f}'}"
            for k in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET))
        is_demo = r.get("source") == corrections.SOURCE_DEMO_SIMULATION
        source_cell = (theme.pill("演示（模拟）", "warn") if is_demo
                       else theme.pill("用户上传", "info"))
        note = (r.get("notes") or "").strip()
        note_cell = (note[:60] + "…") if len(note) > 60 else note
        rows.append(
            f"<tr><td style='white-space:nowrap'>r{r.get('revision')}</td>"
            f"<td>{source_cell}</td>"
            f"<td style='font-size:11.5px;white-space:nowrap'>{dice_txt}</td>"
            f"<td style='font-size:11.5px;color:#66788A'>{note_cell}</td>"
            f"<td style='font-size:11.5px;color:#66788A;white-space:nowrap'>"
            f"{(r.get('created_at') or '')[:19].replace('T', ' ')}</td></tr>")
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px">'
        "<thead><tr style='color:#66788A;font-size:11.5px'>"
        "<th style='text-align:left;white-space:nowrap'>版本</th>"
        "<th style='text-align:left;white-space:nowrap'>来源</th>"
        "<th style='text-align:left;white-space:nowrap'>与预测的 Dice</th>"
        "<th style='text-align:left'>备注</th>"
        "<th style='text-align:left;white-space:nowrap'>时间</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        f'<div class="rc-hint" style="margin-top:6px">{corrections.CORRECTION_DISCLAIMER}</div>')


def _correction_diff_table(diff: corrections.CorrectionDiff) -> str:
    rows = []
    for region, zh in ((metrics.REGION_WT, "全肿瘤区"), (metrics.REGION_TC, "肿瘤核心区"),
                       (metrics.REGION_ET, "强化肿瘤区")):
        d = diff.dice.get(region)
        rows.append(
            f"<tr><td><strong>{region}</strong> · {zh}</td>"
            f"<td style='text-align:right'>{'—' if d is None else f'{d:.4f}'}</td>"
            f"<td style='text-align:right'>{diff.predicted_ml.get(region, 0):.2f}</td>"
            f"<td style='text-align:right'>{diff.corrected_ml.get(region, 0):.2f}</td>"
            f"<td style='text-align:right'>{diff.volume_delta_ml.get(region, 0):+.2f}</td>"
            f"<td style='text-align:right'>{diff.added_voxels.get(region, 0):,}</td>"
            f"<td style='text-align:right'>{diff.removed_voxels.get(region, 0):,}</td></tr>")
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px">'
        "<thead><tr style='color:#66788A;font-size:11.5px'>"
        "<th style='text-align:left'>区域</th><th style='text-align:right'>Dice</th>"
        "<th style='text-align:right'>预测 mL</th><th style='text-align:right'>修正 mL</th>"
        "<th style='text-align:right'>体积变化</th>"
        "<th style='text-align:right'>修正新增</th><th style='text-align:right'>修正删除</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<div class="rc-hint" style="margin-top:6px">'
        "Dice 为模型预测与人工修正掩膜的一致性度量，不代表临床诊断准确性。</div>")


def on_import_correction(session: Session, mask_file, upload_format: str,
                         region: str | None):
    """Validate an uploaded mask and show prediction / correction / difference."""
    case = get_case(session.case_id)
    if case is None:
        return (session, None, theme.banner("请先选择病例。", "warn"), "", "")
    if not session.seg_done:
        return (session, None,
                theme.banner("尚无模型预测结果，无法进行修正比对。请先运行 AI 分割。", "warn"),
                "", "")
    if not mask_file:
        return (session, None,
                theme.banner("请先选择要导入的 NIfTI 掩膜文件。", "warn"), "", "")

    try:
        import nibabel as nib
        img = nib.load(str(mask_file))
        data = np.asarray(img.dataobj)
        uploaded_affine = np.asarray(img.affine, dtype=float)
    except Exception as exc:  # noqa: BLE001
        return (session, None,
                theme.banner(f"掩膜读取失败：{type(exc).__name__}: {exc}", "err"), "", "")

    base = cached_volume(case.image_paths[case.modalities[0]])
    prediction = load_prediction(case.case_id)
    if prediction is None:
        return (session, None,
                theme.banner("未找到已保存的模型预测文件。", "err"), "", "")

    # Geometry first: a mask on a different grid would still yield Dice numbers,
    # and those numbers would be meaningless.
    validation = corrections.validate_correction(
        data[..., 0] if data.ndim == 4 else data,
        expected_shape=tuple(base.data.shape[:3]),
        affine=uploaded_affine,
        expected_affine=base.affine,
        convention=case.convention,
        upload_format=upload_format,
        # Geometry alone cannot catch a mask from a different patient: all BraTS
        # volumes share one grid, so the filename is the only place a mix-up is
        # visible.
        source_name=str(mask_file),
        expected_case_id=case.case_id,
    )
    if not validation.ok:
        return (session, None, theme.banner(
            "掩膜与当前病例不匹配，已拒绝导入。<br>"
            + "<br>".join(f"· {p}" for p in validation.problems), "err"), "", "")

    try:
        parsed, warnings = corrections.parse_uploaded_mask(
            data, upload_format, region, case.convention)
    except ValueError as exc:
        return (session, None, theme.banner(f"掩膜解析失败：{exc}", "err"), "", "")

    diff = corrections.compare_to_prediction(
        prediction, parsed, base.spacing, case.convention)

    session.pending_correction = {
        "label_map": parsed, "validation": validation.to_dict(),
        "diff": diff, "source": corrections.SOURCE_USER_UPLOAD,
        "warnings": warnings,
    }

    figure = research.build_comparison_figure(
        base.data, base.affine, prediction, parsed, case.convention,
        slice_index=session.viewer.slice_index,
        crop=cached_bbox(case.image_paths[case.modalities[0]], base.data, base.affine),
        display_size=DISPLAY_SIZE,
    )
    banner = theme.banner(
        "掩膜校验通过，已生成差异分析。确认无误后点击「保存为科研反馈」写入记录。"
        + ("<br>· " + "；".join(warnings) if warnings else ""), "ok")
    return session, figure, banner, _correction_diff_table(diff), ""


def on_save_correction_feedback(session: Session, error_type: str, notes: str):
    """Persist the pending correction as a new revision plus a research record."""
    case = get_case(session.case_id)
    pending = session.pending_correction
    if case is None or not pending:
        return (session, theme.banner("没有待保存的修正结果，请先导入掩膜。", "warn"),
                correction_history_html(session.case_id), research_dashboard_html(), "")

    label_map = pending["label_map"]
    diff: corrections.CorrectionDiff = pending["diff"]
    source = pending["source"]
    base = cached_volume(case.image_paths[case.modalities[0]])
    revision = STORE.next_correction_revision(case.case_id)

    # Written to both records, so the correction log and the research feedback
    # carry identical provenance text and cannot drift apart.
    note_text = notes.strip() if notes and notes.strip() else corrections.feedback_note(
        diff, source, revision)

    try:
        record = corrections.save_correction(
            label_map, base.affine, root=CORRECTION_DIR_ROOT, case_id=case.case_id,
            revision=revision, source=source,
            validation=pending["validation"], metrics_payload=diff.to_dict(),
            notes=note_text, store=STORE)
    except Exception as exc:  # noqa: BLE001
        return (session, theme.banner(f"保存失败：{type(exc).__name__}: {exc}", "err"),
                correction_history_html(session.case_id), research_dashboard_html(), "")

    # A suggestion only; the reviewer's choice wins when they made one.
    category = error_type or corrections.suggest_error_type_from_diff(diff)
    STORE.record_feedback(
        case.case_id, category,
        dataset_source=case.dataset_source,
        model_name=session.seg_meta.get("model_name") or segmentation.MODEL_NAME,
        model_version=session.seg_meta.get("model_version"),
        notes=note_text,
        metrics={"correction_diff": diff.to_dict(), "correction_revision": revision},
    )

    session.pending_correction = None
    session.last_correction_revision = revision
    label = research.ERROR_TYPE_LABELS.get(category, category)
    return (session,
            theme.banner(
                f"已保存第 {revision} 版人工修正（{record.source_label}），"
                f"并写入科研反馈记录（错误类型：{label}）。原始模型预测未被修改。",
                "ok"),
            correction_history_html(case.case_id), research_dashboard_html(), "")


def on_simulate_correction_from_reference(session: Session):
    """Demo path: replay the dataset reference annotation as if it were a correction.

    Labelled ``demo_reference_simulation`` throughout, so a demo can never be
    read as evidence of real human correction performance.
    """
    case = get_case(session.case_id)
    if case is None:
        return (session, None, theme.banner("请先选择病例。", "warn"), "", "")
    if not case.has_reference:
        return (session, None,
                theme.banner("该病例没有数据集参考标注，无法生成演示修正。", "warn"), "", "")
    if not session.seg_done:
        return (session, None,
                theme.banner("尚无模型预测结果，请先运行 AI 分割。", "warn"), "", "")

    base = cached_volume(case.image_paths[case.modalities[0]])
    prediction = load_prediction(case.case_id)
    ref = cached_volume(str(case.reference_path())).data.astype(np.uint8)

    diff = corrections.compare_to_prediction(prediction, ref, base.spacing, case.convention)
    session.pending_correction = {
        "label_map": ref,
        "validation": {"ok": True, "format": "demo", "note": "来自数据集参考标注"},
        # The CorrectionDiff object, not its dict form -- the save path calls
        # methods on it, and the two entry points must agree on the shape.
        "diff": diff,
        "source": corrections.SOURCE_DEMO_SIMULATION,
        "warnings": [],
    }
    figure = research.build_comparison_figure(
        base.data, base.affine, prediction, ref, case.convention,
        slice_index=session.viewer.slice_index,
        crop=cached_bbox(case.image_paths[case.modalities[0]], base.data, base.affine),
        display_size=DISPLAY_SIZE,
    )
    return (session, figure,
            theme.banner(corrections.DEMO_DISCLAIMER, "warn"),
            _correction_diff_table(diff), "")


def on_build_comparison(session: Session):
    """Render the prediction / reference / difference panel for the case in view.

    Requires a real prediction *and* a real reference annotation; when either is
    missing the caller gets an explanation rather than a placeholder figure.
    """
    case = get_case(session.case_id)
    if case is None:
        return None, theme.empty_state("target", "请先选择病例",
                                       "误差对比图需要当前病例的真实预测与参考标注")

    pred = load_prediction(case.case_id) if session.seg_done else None
    ref_path = case.reference_path()
    ref = cached_volume(str(ref_path)).data.astype(np.uint8) if ref_path else None

    caption = research.comparison_caption(pred is not None, ref is not None)
    if caption:
        return None, theme.banner(caption, "warn")

    base = cached_volume(case.image_paths[case.modalities[0]])
    figure = research.build_comparison_figure(
        base.data, base.affine, pred, ref, case.convention,
        slice_index=session.viewer.slice_index,
        crop=cached_bbox(case.image_paths[case.modalities[0]], base.data, base.affine),
        display_size=DISPLAY_SIZE,
    )
    ev = session.evaluation or {}
    dice = ev.get("dice", {})
    dice_txt = " · ".join(
        f"{r} {dice[r]:.4f}" for r in (metrics.REGION_WT, metrics.REGION_TC, metrics.REGION_ET)
        if isinstance(dice.get(r), (int, float))
    )
    head = theme.banner(
        f"左：模型预测　中：数据集参考标注　右：差异（紫=漏分割，黄=过度分割）"
        + (f"　|　Dice：{dice_txt}" if dice_txt else ""),
        "info")
    return figure, head + research.error_legend_html()


def feedback_table_html() -> str:
    records = STORE.list_feedback(limit=50)
    if not records:
        return theme.empty_state("microscope", "暂无科研反馈记录", "请在下方填写错误分类并保存")
    rows = []
    for r in records:
        label = research.ERROR_TYPE_LABELS.get(r.get("error_type"), r.get("error_type"))
        rows.append(
            f"<tr><td>{r.get('case_id','')}</td><td>{label}</td>"
            f"<td style='font-size:11.5px;color:#66788A'>{r.get('model_name') or '—'}</td>"
            f"<td style='font-size:11.5px;color:#66788A'>{(r.get('notes') or '')[:40]}</td>"
            f"<td style='font-size:11.5px;color:#66788A'>{(r.get('created_at') or '')[:19]}</td></tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px">'
        "<thead><tr style='color:#66788A;font-size:11.5px'>"
        "<th style='text-align:left'>病例</th><th style='text-align:left'>错误类型</th>"
        "<th style='text-align:left'>模型</th><th style='text-align:left'>备注</th>"
        "<th style='text-align:left'>时间</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def on_export_csv(kind: str):
    if kind == "评测指标":
        path = research.write_export(research.evaluation_csv(STORE), EXPORT_DIR, "evaluation")
    else:
        path = research.write_export(research.feedback_csv(STORE.list_feedback()), EXPORT_DIR, "feedback")
    return path, theme.banner(f"已导出 <code>{Path(path).name}</code>", "ok")


# ==========================================================================
# Dashboard
# ==========================================================================
def dashboard_status_html(session: Session) -> str:
    case = get_case(session.case_id)
    counts = STORE.counts()
    n_series = len(case.modalities) if case else None
    model = segmentation.model_status()
    last_run = ""
    if session.seg_meta:
        last_run = f"{session.seg_meta.get('duration_s', 0):.1f}s · {session.seg_meta.get('device','')}"
    elif case is not None:
        rows = STORE.list_analysis_runs(case.case_id, limit=1)
        if rows:
            last_run = f"{rows[0].get('duration_s','')}s · {rows[0].get('device','')}"

    review = {"unreviewed": "未审核", "draft": "草稿", "pending_review": "待审核",
              "approved": "已审核"}.get(session.review_status, session.review_status)

    items = [
        theme.stat("当前病例", case.case_id if case else None,
                   case.origin_label if case else ""),
        theme.stat("已加载 MRI 序列", f"{n_series} 个" if n_series else None,
                   "、".join(imaging.MODALITY_LABELS.get(m, m.upper()) for m in case.modalities) if case else ""),
        theme.stat("当前分割模型",
                   f"{model['architecture']}" if model["available"] else "未就绪",
                   f"{model['model_name']} v{model['model_version']}" if model["available"]
                   else "运行 scripts/fetch_model.py 下载权重"),
        theme.stat("最近一次分析", last_run or None,
                   "已完成模型推理" if last_run else "尚未运行"),
        theme.stat("报告审核状态", review if session.report_content else None,
                   "报告草稿已生成" if session.report_content else "尚未生成报告"),
        theme.stat("科研反馈记录", f"{counts['feedback']} 条" if counts["feedback"] else None,
                   f"含 {counts['analyses']} 例模型评测" if counts["analyses"] else "尚无评测记录"),
    ]
    return f'<div class="rc-status-grid">{"".join(items)}</div>'


def build_dashboard(session: Session) -> str:
    reg = REGISTRY
    n_real = len(reg.real_cases())
    data_note = (
        f'当前已就绪：<strong>{n_real} 例真实公开 BraTS 2020 病例</strong>'
        f"（含专家参考标注）" if n_real else
        "当前无真实病例数据，请运行 <code>python scripts/fetch_data.py</code> 获取"
    )
    return f"""
<div class="rc-hero">
  <h1>Radiology Copilot</h1>
  <p>诊教研一体化医学影像 AI 工作台 · 面向影像科医生、医学影像学生与科研人员，
  把 AI 分割、结构化报告、教学练习与科研反馈串成一个围绕病例的闭环工作流。
  第一版聚焦<strong>脑肿瘤 MRI</strong>。</p>
</div>

<div class="rc-features">
  <div class="rc-feature">
    <div class="rc-feature-icon">{theme.icon("brain", 18, theme.BLUE)}</div>
    <h3>智能阅片</h3>
    <p>四序列 MRI 浏览、AI 病灶分割、区域体积定量与参考标注对比评测。</p>
  </div>
  <div class="rc-feature">
    <div class="rc-feature-icon">{theme.icon("report", 18, theme.TEAL)}</div>
    <h3>AI 报告</h3>
    <p>基于已确认的分割与测量结果，生成可编辑、可审核、可导出的结构化报告草稿。</p>
  </div>
  <div class="rc-feature">
    <div class="rc-feature-icon">{theme.icon("teaching", 18, "#7A5AF8")}</div>
    <h3>智能教学</h3>
    <p>隐藏答案的病例练习、学生掩膜上传统分、与参考标注的一致性反馈。</p>
  </div>
  <div class="rc-feature">
    <div class="rc-feature-icon">{theme.icon("research", 18, "#E58A2E")}</div>
    <h3>科研工作台</h3>
    <p>模型评测统计、错误分类、困难病例清单与反馈记录导出。</p>
  </div>
</div>

<div style="height:14px"></div>
{theme.card("当前状态", dashboard_status_html(session))}
{theme.banner(data_note + "。本工作台不会显示虚构的患者数量、准确率或使用量数据；无数据时统一显示空状态。", "info")}
"""


# ==========================================================================
# UI assembly
# ==========================================================================
def _initial_view() -> dict[str, Any]:
    """Compute the state the app opens with.

    The first case is pre-loaded rather than leaving an empty viewer behind a
    dropdown, so the workspace is usable the moment the page renders.
    """
    first = (REGISTRY.ids() or [None])[0]
    session = Session()
    if first is None:
        return {
            "session": session, "case_id": None, "image": _empty_figure("请先选择病例"),
            "max_slice": 154, "slice": 77, "info": case_info_html(session),
            "meta": "—",
            "status": theme.banner("尚未发现任何病例数据。请参考 README 获取公开数据集。", "warn"),
            "measured": theme.empty_state("ruler", "暂无测量数据", "请先选择病例"),
            "evaluated": theme.empty_state("target", "暂无评测结果", "请先选择病例"),
            "topbar": theme.topbar(None, *_model_badge()),
        }

    # Reuse the real callback so the startup path cannot drift from the
    # interactive path.
    (_s, image, slider, info, meta, status, measured, evaluated,
     topbar_html) = on_case_selected(first, session)
    case = get_case(first)
    vol = cached_volume(case.image_paths[case.modalities[0]])
    return {
        "session": session, "case_id": first, "image": image,
        "max_slice": imaging.axial_slice_count(vol.data, vol.affine) - 1,
        "slice": session.viewer.slice_index,
        "info": info,
        "meta": meta,
        "status": status,
        "measured": measured,
        "evaluated": evaluated,
        "topbar": topbar_html,
    }


def build_app() -> gr.Blocks:
    init = _initial_view()

    with gr.Blocks(css=theme.CSS, title="Radiology Copilot · 影像智能诊教研工作台",
                   theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        session = gr.State(init["session"])
        topbar = gr.HTML(init["topbar"])

        with gr.Row():
            # ------------------------------------------------ left nav ---
            with gr.Column(scale=1, min_width=172, elem_classes="rc-nav"):
                nav_buttons = {}
                for key, (label, _icon) in {
                    "dashboard": ("工作台", None),
                    "viewer": ("智能阅片", None),
                    "report": ("AI 报告", None),
                    "teaching": ("教学模式", None),
                    "research": ("科研反馈", None),
                }.items():
                    nav_buttons[key] = gr.Button(label, elem_classes=["rc-nav-btn"], size="sm")

            # --------------------------------------------- main content ---
            with gr.Column(scale=6):
                # ---- page: dashboard ---------------------------------
                with gr.Column(visible=True) as page_dashboard:
                    dashboard_html = gr.HTML(build_dashboard(init["session"]))
                    refresh_dash = gr.Button("刷新状态", variant="secondary", size="sm")

                # ---- page: viewer ------------------------------------
                with gr.Column(visible=False) as page_viewer:
                    with gr.Row():
                        case_dd = gr.Dropdown(
                            choices=case_choices(), value=init["case_id"],
                            label="选择病例", interactive=True, scale=4)
                        run_btn = gr.Button("▶ 运行 AI 分割", variant="primary", scale=1)
                        ref_btn = gr.Button("参考标注模式", variant="secondary", scale=1)

                    status_html = gr.HTML()
                    case_status = gr.HTML(init["status"])

                    with gr.Row():
                        # left: compact case identity; technical detail folded away
                        with gr.Column(scale=2, min_width=190):
                            info_html = gr.HTML(init["info"])
                            with gr.Accordion("技术信息", open=False):
                                meta_html = gr.HTML(init["meta"])
                        # centre: the image is the point of the page, so it gets
                        # the widest column and the tallest panel
                        with gr.Column(scale=7, min_width=460, elem_classes="rc-viewer"):
                            mri_image = gr.Image(value=init["image"],
                                                 label="", show_label=False, type="numpy",
                                                 interactive=False, height=500)
                            slice_info = gr.HTML("")
                            legend_html = gr.HTML("")
                        # right: controls + measurements
                        with gr.Column(scale=3, min_width=255):
                            modality_rg = gr.Radio(
                                choices=[(imaging.MODALITY_LABELS[m], m) for m in imaging.MODALITIES],
                                value="t1ce", label="MRI 序列")
                            slice_sl = gr.Slider(0, init["max_slice"], value=init["slice"],
                                                 step=1, label="轴位层面")
                            show_mask_cb = gr.Checkbox(value=True, label="显示分割掩膜")
                            alpha_sl = gr.Slider(0.0, 0.9, value=0.45, step=0.05, label="掩膜透明度")
                            measured = gr.HTML(init["measured"])
                            evaluated = gr.HTML(init["evaluated"])
                            with gr.Accordion("灰度窗调整", open=False):
                                win_low = gr.Slider(0, 20, value=1.0, step=0.5, label="下限百分位")
                                win_high = gr.Slider(80, 100, value=99.0, step=0.5, label="上限百分位")

                # ---- page: report ------------------------------------
                with gr.Column(visible=False) as page_report:
                    gr.HTML(theme.card(
                        "AI 报告助手",
                        '<div class="rc-hint">通过下方助手获取真实数据，或直接生成结构化报告草稿。'
                        '报告默认状态为「待审核」，需人工确认后才会标记为已审核。</div>'))
                    with gr.Row(equal_height=False):
                        # -- left: assistant.  Chat and tool trace scroll
                        #    independently so neither pushes the other out of view.
                        with gr.Column(scale=4, min_width=340):
                            llm_picker = gr.Radio(
                                choices=[(agent_mod.PROVIDER_LABELS[k], k) for k in
                                         (agent_mod.PROVIDER_AUTO, agent_mod.PROVIDER_LOCAL,
                                          agent_mod.PROVIDER_DEEPSEEK)],
                                value=agent_mod.DEFAULT_PROVIDER, label="推理模型",
                                elem_classes="rc-llm-picker")
                            chatbot = gr.Chatbot(label="分析助手", height=186, type="messages",
                                                 elem_classes="rc-chat")
                            with gr.Row():
                                msg_box = gr.Textbox(
                                    placeholder="例如：分析当前病例 / 显示肿瘤区域体积",
                                    show_label=False, scale=5, lines=1)
                                send_btn = gr.Button("发送", variant="primary", scale=1)
                            mode_html = gr.HTML("")
                            gr.HTML('<div class="rc-view-label" style="margin-top:4px">工具调用记录</div>')
                            trace_html = gr.HTML(theme.trace_html([]),
                                                 elem_classes="rc-trace-scroll")
                            gr.HTML('<div class="rc-view-label" style="margin-top:8px">示例问题</div>')
                            with gr.Row():
                                quick_btns = [
                                    gr.Button(q, size="sm", variant="secondary")
                                    for q in ("分析当前病例", "显示肿瘤区域体积",
                                              "当前病例有哪些模型评估结果？")
                                ]
                        # -- right: the report itself gets the wider column
                        with gr.Column(scale=6, min_width=420):
                            with gr.Row():
                                gen_tpl_btn = gr.Button("生成报告草稿（模板）", variant="primary", size="sm")
                                gen_llm_btn = gr.Button("生成报告草稿（LLM 叙述）", variant="secondary", size="sm")
                            gen_note = gr.HTML("")
                            with gr.Tabs():
                                with gr.Tab("预览"):
                                    report_preview = gr.Markdown(
                                        "_尚未生成报告。点击上方按钮生成报告草稿。_",
                                        elem_classes="rc-report-preview")
                                with gr.Tab("编辑"):
                                    report_box = gr.Textbox(
                                        label="", show_label=False, lines=22, max_lines=30,
                                        placeholder="报告正文（Markdown），可直接修改后确认审核。")
                            with gr.Row():
                                approve_btn = gr.Button("确认审核", variant="primary")
                                export_btn = gr.Button("导出报告", variant="secondary")
                            report_file = gr.File(label="下载", interactive=False)
                            gen_by = gr.Textbox(label="生成方式", interactive=False, visible=False)

                # ---- page: teaching ----------------------------------
                with gr.Column(visible=False) as page_teaching:
                    with gr.Row():
                        teach_dd = gr.Dropdown(choices=case_choices(),
                                               value=init["case_id"],
                                               label="教学病例", interactive=True, scale=4)
                        teach_load = gr.Button("加载练习", variant="primary", scale=1)
                        teach_reveal = gr.Button("显示参考标注", variant="secondary", scale=1)
                    teach_status = gr.HTML(theme.banner(
                        "载入病例后，AI 结果与参考标注将默认隐藏。提交答案后方可查看。", "info"))
                    with gr.Row():
                        with gr.Column(scale=2, min_width=230):
                            teach_task = gr.HTML(theme.empty_state("teaching", "尚未加载练习", "请选择病例并点击「加载练习」"))
                        with gr.Column(scale=5, elem_classes="rc-viewer"):
                            teach_image = gr.Image(value=_empty_figure("请选择教学病例"),
                                                   show_label=False, type="numpy",
                                                   interactive=False, height=430)
                        with gr.Column(scale=3, min_width=260):
                            with gr.Tabs():
                                with gr.Tab("答题与提交"):
                                    answer_box = gr.Textbox(
                                        label="你的影像表现记录",
                                        placeholder="描述观察到的异常区域、信号特点、累及范围……",
                                        lines=5)
                                    mask_upload = gr.File(
                                        label="上传分割掩膜（可选，.nii / .nii.gz）",
                                        file_types=[".nii", ".nii.gz"], type="filepath")
                                    submit_btn = gr.Button("提交答案", variant="primary")
                                with gr.Tab("反馈结果"):
                                    # Must be Markdown, not HTML: the feedback
                                    # is markdown and gr.HTML would collapse
                                    # every newline into a single paragraph.
                                    feedback_md = gr.Markdown(
                                        "**参考答案已隐藏** — 提交答案后显示反馈与对比。")

                # ---- page: research ----------------------------------
                with gr.Column(visible=False) as page_research:
                    research_panel = gr.HTML(research_dashboard_html())

                    with gr.Accordion("误差可视化（预测 / 参考 / 差异）", open=False):
                        gr.HTML('<div class="rc-hint">'
                                '基于<strong>当前病例</strong>的真实模型预测与数据集参考标注生成，'
                                '共享同一层面与同一裁剪框。缺任一数据源时不会生成替代图。</div>')
                        with gr.Row():
                            cmp_btn = gr.Button("生成误差对比图", variant="primary", size="sm")
                        cmp_image = gr.Image(show_label=False, type="numpy", interactive=False,
                                             height=300, elem_classes="rc-viewer")
                        cmp_note = gr.HTML(theme.empty_state(
                            "target", "尚未生成对比图",
                            "需要当前病例同时具备模型预测与参考标注"))

                    with gr.Accordion("人工修正掩膜回流", open=False):
                        gr.HTML(
                            '<div class="rc-hint">'
                            '导入由 3D Slicer 等工具修改后的 NIfTI 掩膜，系统会校验尺寸、'
                            'affine、空间方向与标签编码，并给出与模型预测的差异分析。'
                            '原始预测文件<strong>不会被覆盖</strong>，每次修正保存为独立修订版本。<br>'
                            f'<strong>注意</strong>：{corrections.CORRECTION_DISCLAIMER}'
                            '</div>')
                        with gr.Row():
                            corr_file = gr.File(
                                label="修正掩膜 (.nii / .nii.gz)",
                                file_types=[".nii", ".nii.gz"], type="filepath", scale=3)
                            corr_format = gr.Radio(
                                choices=[(corrections.FORMAT_LABELS[k], k)
                                         for k in (corrections.FORMAT_BRATS_LABELMAP,
                                                   corrections.FORMAT_BINARY_REGION)],
                                value=corrections.FORMAT_BRATS_LABELMAP,
                                label="掩膜格式", scale=3)
                            corr_region = gr.Dropdown(
                                choices=[("全肿瘤区 WT", metrics.REGION_WT),
                                         ("肿瘤核心区 TC", metrics.REGION_TC),
                                         ("强化肿瘤区 ET", metrics.REGION_ET)],
                                value=metrics.REGION_WT, label="二值掩膜区域", scale=2)
                        with gr.Row():
                            corr_import_btn = gr.Button("校验并比对", variant="primary", size="sm")
                            corr_demo_btn = gr.Button("演示：用参考标注模拟",
                                                      variant="secondary", size="sm")
                        corr_status = gr.HTML(theme.empty_state(
                            "upload", "尚未导入修正掩膜",
                            "校验通过后可查看「原始预测 / 修正结果 / 差异区域」三联图"))
                        corr_image = gr.Image(show_label=False, type="numpy", interactive=False,
                                              height=290, elem_classes="rc-viewer")
                        corr_table = gr.HTML("")
                        with gr.Row():
                            corr_err = gr.Radio(
                                choices=[(label, key) for key, label in research.ERROR_TYPES],
                                value=None, label="错误类型（留空则按差异自动建议）", scale=4)
                            corr_notes = gr.Textbox(
                                label="修正备注", lines=1, scale=4,
                                placeholder="例如：水肿区边界手工外扩，参考 FLAIR 高信号范围")
                        corr_save_btn = gr.Button("保存为科研反馈", variant="primary", size="sm")
                        corr_save_note = gr.HTML("")

                        gr.HTML('<div class="rc-view-label" style="margin-top:10px">修正历史</div>')
                        corr_history = gr.HTML(correction_history_html(init["case_id"]))

                    with gr.Row():
                        refresh_research = gr.Button("刷新统计", variant="secondary", size="sm")
                        csv_kind = gr.Radio(choices=["评测指标", "科研反馈"], value="评测指标",
                                            label="导出内容", scale=2)
                        csv_btn = gr.Button("导出 CSV", variant="secondary", size="sm")
                    csv_file = gr.File(label="下载", interactive=False)
                    csv_note = gr.HTML("")

                    gr.HTML(theme.card(
                        "记录科研反馈",
                        '<div class="rc-hint">针对当前病例记录模型偏差。错误类型由你确认后写入数据库，'
                        '系统只提供基于体素误差的预选建议。</div>'))
                    with gr.Row():
                        err_type = gr.Radio(
                            choices=[(label, key) for key, label in research.ERROR_TYPES],
                            value=None, label="错误类型", scale=3)
                        fb_notes = gr.Textbox(label="备注", placeholder="例如：水肿区边界明显欠分割，需结合 FLAIR 复核",
                                              lines=2, scale=4)
                    save_fb_btn = gr.Button("保存反馈记录", variant="primary")
                    fb_note = gr.HTML("")
                    fb_table = gr.HTML(feedback_table_html())

        gr.HTML(f'<div class="rc-footer">{DISCLAIMER_FOOTER}</div>')

        # ------------------------------------------------------ wiring ---
        PAGE_ORDER = ["dashboard", "viewer", "report", "teaching", "research"]

        def make_switch(target: str):
            def _switch():
                active = {k: (k == target) for k in PAGE_ORDER}
                classes = [
                    ["rc-nav-btn", "rc-nav-btn--active"] if k == target else ["rc-nav-btn"]
                    for k in PAGE_ORDER
                ]
                return (
                    [gr.update(visible=active[k]) for k in PAGE_ORDER]
                    + [gr.update(elem_classes=c) for c in classes]
                )
            return _switch

        page_cols = [page_dashboard, page_viewer, page_report, page_teaching, page_research]
        for key, btn in nav_buttons.items():
            btn.click(make_switch(key), None, page_cols + list(nav_buttons.values()))

        # The research panel and the dashboard are built from the database, so
        # they must be re-read when the user navigates to them -- otherwise they
        # keep showing whatever was true when the app started.
        nav_buttons["research"].click(
            lambda: (research_dashboard_html(), feedback_table_html()),
            None, [research_panel, fb_table])
        nav_buttons["dashboard"].click(
            lambda s: build_dashboard(s), [session], [dashboard_html])

        viewer_outputs = [session, mri_image, slice_sl, info_html, meta_html,
                          case_status, measured, evaluated, topbar]
        case_dd.change(on_case_selected, [case_dd, session], viewer_outputs).then(
            # The correction history is per case, so it must not carry over.
            lambda s: (correction_history_html(s.case_id), "", None, ""),
            [session], [corr_history, corr_table, corr_image, corr_status])
        viewer_outputs2 = [session, mri_image, slice_info, legend_html]
        for comp in (modality_rg, slice_sl, show_mask_cb, alpha_sl, win_low, win_high):
            comp.change(on_view_change, [session, modality_rg, slice_sl, show_mask_cb,
                                         alpha_sl, win_low, win_high], viewer_outputs2)
        run_btn.click(on_run_segmentation, [session],
                      [session, mri_image, case_status, measured, evaluated, legend_html]
                      ).then(lambda: (research_dashboard_html(), feedback_table_html()),
                             None, [research_panel, fb_table])
        ref_btn.click(on_use_reference_mode, [session],
                      [session, mri_image, case_status, measured, evaluated, legend_html])

        refresh_dash.click(lambda s: build_dashboard(s), [session], [dashboard_html])

        send_btn.click(on_agent_message, [msg_box, chatbot, session],
                       [chatbot, session, trace_html, mode_html]).then(
            lambda: "", None, msg_box)
        msg_box.submit(on_agent_message, [msg_box, chatbot, session],
                       [chatbot, session, trace_html, mode_html]).then(
            lambda: "", None, msg_box)

        report_outputs = [session, report_box, report_preview, gen_note, gen_by]
        gen_tpl_btn.click(lambda s: on_generate_report(s, False), [session], report_outputs)
        gen_llm_btn.click(lambda s: on_generate_report(s, True), [session], report_outputs)
        approve_btn.click(on_approve_report, [session, report_box],
                          [session, gen_note, report_box, report_preview])
        export_btn.click(on_export_report, [session, report_box], [report_file, gen_note])
        # Keep the preview in step with manual edits in the 编辑 tab.
        report_box.change(lambda text: text or "_报告内容为空。_", [report_box], [report_preview])

        teach_load.click(on_teaching_load, [session, teach_dd],
                         [session, teach_image, teach_task, answer_box, feedback_md,
                          teach_status, mask_upload])
        submit_btn.click(on_teaching_submit, [session, answer_box, mask_upload],
                         [session, feedback_md, teach_status])
        teach_reveal.click(on_teaching_reveal, [session], [session, teach_image, teach_status])

        for btn, question in zip(quick_btns, ("分析当前病例", "显示肿瘤区域体积",
                                              "当前病例有哪些模型评估结果？")):
            btn.click(lambda q=question: q, None, msg_box)

        save_fb_btn.click(on_save_feedback, [session, err_type, fb_notes],
                          [session, fb_note, fb_table, research_panel])
        refresh_research.click(lambda: (research_dashboard_html(), feedback_table_html()),
                               None, [research_panel, fb_table])
        csv_btn.click(on_export_csv, [csv_kind], [csv_file, csv_note])
        cmp_btn.click(on_build_comparison, [session], [cmp_image, cmp_note])
        llm_picker.change(on_llm_choice_changed, [session, llm_picker],
                          [session, mode_html])

        corr_outputs = [session, corr_image, corr_status, corr_table, corr_save_note]
        corr_import_btn.click(on_import_correction,
                              [session, corr_file, corr_format, corr_region], corr_outputs)
        corr_demo_btn.click(on_simulate_correction_from_reference, [session], corr_outputs)
        corr_save_btn.click(on_save_correction_feedback,
                            [session, corr_err, corr_notes],
                            [session, corr_save_note, corr_history, research_panel, corr_status])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; see README for SSH forwarding)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    print("=" * 74)
    print("  Radiology Copilot · 影像智能诊教研工作台")
    print("  仅供科研与教学演示，不用于临床诊断")
    print("=" * 74)
    print(f"  病例数据     : {REGISTRY.describe()}")
    if REGISTRY.warnings:
        for w in REGISTRY.warnings:
            print(f"  [警告] {w}")
    model = segmentation.model_status()
    print(f"  分割模型     : {model['model_name']} v{model['model_version']} "
          f"({model['architecture']})")
    print(f"  权重状态     : {'已就绪' if model['available'] else '未下载 —— 将使用参考标注模式'}")
    print(f"  推理设备     : {model['device']}  ({model['device_reason']})")
    cfg = agent_mod.load_llm_config()
    print(f"  LLM Agent    : {cfg.describe()}")
    print("-" * 74)
    print(f"  访问地址     : http://{args.host}:{args.port}")
    print("  远程服务器请使用 SSH 端口转发：")
    print(f"      ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>")
    print("=" * 74)

    build_app().launch(server_name=args.host, server_port=args.port,
                       share=args.share, show_api=False, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
