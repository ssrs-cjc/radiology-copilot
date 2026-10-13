"""Teaching mode: a case exercise with the answer withheld until submission.

The integrity rule for this module is simple and absolute: **the reference
answer must not reach the UI before the student submits**.  Everything here is
built around that, including :class:`ExerciseState` which refuses to hand out
reference data while ``submitted`` is false.

The second rule is honesty about what can be graded.  A Dice score against a
reference mask is objective and is reported as such.  A free-text answer is
*not* automatically gradable against a clinical ground truth -- this prototype
has no such ground truth and no validated rubric -- so free-text feedback is
limited to a checklist of factual points drawn from the case metadata, and
anything requiring clinical judgement is explicitly marked as not auto-graded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import metrics
from metrics import REGION_ET, REGION_TC, REGION_WT

# Thresholds used to phrase feedback.  These describe agreement with a single
# reference annotation, not clinical correctness.
DICE_STRONG = 0.70
DICE_MODERATE = 0.50
DICE_WEAK = 0.30

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass
class ExerciseState:
    """Per-session state for one exercise attempt.

    The reference mask is held here but :meth:`reference_for_display` refuses to
    return it before submission, so a UI bug cannot leak the answer.

    ``submission_count`` makes repeat submissions explicit: a student may revise
    and resubmit, but each submission is recorded and the answer stays hidden
    until the first one happens.
    """

    case_id: str
    submitted: bool = False
    answer_text: str = ""
    submitted_mask_path: str | None = None
    reference_label_map: np.ndarray | None = None
    reference_convention: Any = None
    reference_spacing: tuple[float, float, float] | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    submission_count: int = 0

    # -- answer withholding ---------------------------------------------
    def reference_for_display(self) -> np.ndarray | None:
        """Reference mask, but only after the student has submitted."""
        if not self.submitted:
            return None
        return self.reference_label_map

    @property
    def answer_visible(self) -> bool:
        return self.submitted

    def record_submission(self, answer_text: str, mask_path: str | None,
                          evaluation: dict[str, Any], feedback: str) -> int:
        """Apply a submission (first or repeated). Returns the new count."""
        self.answer_text = answer_text
        self.submitted_mask_path = mask_path
        self.evaluation = evaluation
        self.feedback = feedback
        self.submitted = True
        self.submission_count += 1
        return self.submission_count

    def reset(self) -> None:
        self.submitted = False
        self.answer_text = ""
        self.submitted_mask_path = None
        self.evaluation = {}
        self.feedback = ""
        self.submission_count = 0


@dataclass(frozen=True)
class RevealDecision:
    """Outcome of the server-side check that gates the reference overlay."""

    allowed: bool
    message: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def reveal_allowed(state: ExerciseState | None) -> RevealDecision:
    """Whether the reference answer may be drawn right now.

    This is the authoritative check, and every render path goes through it.
    Hiding the reveal button in the UI is not sufficient: a client could call
    the callback directly, so the decision is made here, on the server, from the
    session's own submission state.
    """
    if state is None:
        return RevealDecision(False, "尚未加载教学病例，请先点击「加载练习」。")
    if not state.submitted:
        return RevealDecision(
            False,
            "参考答案在提交前不可查看。请先独立完成作答并点击「提交答案」。",
        )
    return RevealDecision(True)


# --------------------------------------------------------------------------
# Mask submission scoring
# --------------------------------------------------------------------------
def score_mask_submission(
    submitted_label_map: np.ndarray,
    reference_label_map: np.ndarray,
    spacing: tuple[float, float, float],
    convention: metrics.LabelConvention,
) -> dict[str, Any]:
    """Score an uploaded mask against the reference by region Dice and volume.

    The uploaded mask is expected to use the same label encoding as the
    reference.  If it does not match, that is reported rather than coerced.
    """
    sub = np.asarray(submitted_label_map)
    ref = np.asarray(reference_label_map)
    if sub.shape != ref.shape:
        return {
            "comparable": False,
            "reason": (
                f"提交的掩膜尺寸 {tuple(sub.shape)} 与参考标注 "
                f"{tuple(ref.shape)} 不一致，无法直接比较。"
                "请在与影像相同的体素网格上提交掩膜。"
            ),
        }

    check = metrics.verify_convention(sub, convention)
    ev = metrics.evaluate_segmentation(sub, ref, spacing, convention)

    sub_masks = metrics.region_masks(sub, convention)
    ref_masks = metrics.region_masks(ref, convention)
    sub_vols = metrics.all_region_volumes(sub_masks, spacing)
    ref_vols = metrics.all_region_volumes(ref_masks, spacing)

    regions: dict[str, Any] = {}
    for region in (REGION_WT, REGION_TC, REGION_ET):
        d = ev.dice[region]
        regions[region] = {
            "dice": d.dice,
            "intersection_voxels": d.intersection,
            "submitted_voxels": d.pred_voxels,
            "reference_voxels": d.ref_voxels,
            "submitted_ml": sub_vols[region].volume_ml,
            "reference_ml": ref_vols[region].volume_ml,
            "volume_delta_ml": sub_vols[region].volume_ml - ref_vols[region].volume_ml,
            "missed_voxels": ev.missed[region],
            "extra_voxels": ev.extra[region],
            "comment": _region_comment(region, d.dice, ev.missed[region], ev.extra[region]),
        }

    return {
        "comparable": True,
        "scored": True,
        "label_check_ok": check.ok,
        "label_check": check.summary(),
        "observed_labels": list(check.observed_labels),
        "mean_dice": ev.mean_dice(),
        "regions": regions,
    }


def _region_comment(region: str, d: float | None, missed: int, extra: int) -> str:
    if d is None:
        return "无法比较。"
    if d >= DICE_STRONG:
        quality = "与参考标注高度一致"
    elif d >= DICE_MODERATE:
        quality = "与参考标注基本一致"
    elif d >= DICE_WEAK:
        quality = "与参考标注存在明显差异"
    else:
        quality = "与参考标注差异很大"

    detail = ""
    if missed > 0 and extra > 0:
        detail = f"其中漏分割 {missed:,} 体素，过度分割 {extra:,} 体素。"
    elif missed > 0:
        detail = f"主要表现为漏分割（{missed:,} 体素）。"
    elif extra > 0:
        detail = f"主要表现为过度分割（{extra:,} 体素）。"
    else:
        detail = "未发现体素级差异。"
    return f"{region} Dice = {d:.4f}，{quality}。{detail}"


# --------------------------------------------------------------------------
# Text submission feedback
# --------------------------------------------------------------------------
# Terms we can check for because they follow directly from the case metadata.
MODALITY_TERMS = {
    "flair": ["flair", "压水", "液体衰减"],
    "t1ce": ["t1ce", "t1c", "增强", "强化", "增强扫描", "对比剂"],
    "t1": ["t1"],
    "t2": ["t2"],
}
REGION_TERMS = {
    "WT": ["全肿瘤", "瘤周水肿", "水肿", "wt", "whole tumor"],
    "TC": ["肿瘤核心", "核心", "坏死", "非强化", "tc", "core"],
    "ET": ["强化", "增强区", "et", "enhancing"],
}


CLINICAL_JUDGEMENT_NOTE = (
    "以下问题涉及临床诊断判断，本系统没有可靠的标准答案，**不进行自动评分**，"
    "请由带教老师人工评阅：肿瘤病理类型、WHO 分级、分子分型、"
    "鉴别诊断、治疗方案与预后判断。"
)


def review_text_answer(answer_text: str, case_meta: dict[str, Any],
                       evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provide a factual checklist for a free-text answer.

    This is explicitly **not** a score.  It checks whether the student used the
    terminology and modality/region facts that the case metadata actually
    supports, and it declines to grade anything clinical.
    """
    text = (answer_text or "").strip()
    if not text:
        return {"checklist": [], "not_graded": CLINICAL_JUDGEMENT_NOTE,
                "checked_terms": 0, "matched_terms": 0}

    lowered = text.lower()
    checklist: list[dict[str, str]] = []

    # 1. Did the answer mention the modalities actually available?
    available = case_meta.get("modalities", [])
    for mod in available:
        terms = MODALITY_TERMS.get(mod, [mod])
        hit = any(t in lowered for t in terms)
        checklist.append({
            "item": f"提及 {mod.upper()} 序列",
            "status": "hit" if hit else "miss",
            "detail": "已提及该序列。" if hit else "未提及该序列，可结合该序列的影像表现补充。",
        })

    # 2. Did the answer describe the three sub-regions?
    for region, terms in REGION_TERMS.items():
        hit = any(t in lowered for t in terms)
        checklist.append({
            "item": f"描述 {region} 区域",
            "status": "hit" if hit else "miss",
            "detail": "已描述该区域。" if hit else f"未描述 {region}，建议补充该区域的影像表现。",
        })

    # 3. If a mask was also submitted, cross-reference the numbers.
    if evaluation and evaluation.get("comparable"):
        wt = evaluation["regions"].get(REGION_WT, {})
        if wt.get("dice") is not None:
            checklist.append({
                "item": "分割结果与参考标注的一致性",
                "status": "hit" if wt["dice"] >= DICE_MODERATE else "info",
                "detail": f"WT Dice = {wt['dice']:.4f}（客观指标，非临床正确性判断）。",
            })

    # 4. Flag if the answer asserts a diagnosis, which we cannot verify.
    diagnosis_pattern = re.compile(
        r"(胶质母细胞瘤|星形细胞瘤|少突|转移瘤|淋巴瘤|脑膜瘤|WHO\s*[IV]+|"
        r"级别|分级|恶性|良性|建议手术|建议放疗|建议化疗)",
        re.IGNORECASE,
    )
    if diagnosis_pattern.search(text):
        checklist.append({
            "item": "涉及诊断或治疗建议的表述",
            "status": "info",
            "detail": "答案包含诊断分级或治疗相关表述，本系统不对此类内容自动评分。",
        })

    matched = sum(1 for c in checklist if c["status"] == "hit")
    return {
        "checklist": checklist,
        "checked_terms": len(checklist),
        "matched_terms": matched,
        "not_graded": CLINICAL_JUDGEMENT_NOTE,
        "disclaimer": (
            "以上为结构化信息核对清单，用于提示答案覆盖度，**不是分数**，"
            "也不代表答案的临床正确性。"
        ),
    }


# --------------------------------------------------------------------------
# Formatting for the UI
# --------------------------------------------------------------------------
def feedback_markdown(evaluation: dict[str, Any], text_review: dict[str, Any] | None,
                      reference_available: bool) -> str:
    lines: list[str] = ["### 练习反馈", ""]

    if not reference_available:
        lines.append(
            "> 该病例没有参考标注，**无法计算分割精度指标**。以下仅为文字答案的结构化核对。"
        )
        lines.append("")

    if evaluation.get("comparable"):
        mean_dice = evaluation.get("mean_dice")
        if mean_dice is not None:
            kind = ("高度一致" if mean_dice >= DICE_STRONG else
                    "基本一致" if mean_dice >= DICE_MODERATE else
                    "存在明显差异" if mean_dice >= DICE_WEAK else "差异很大")
            lines.append(f"**平均 Dice：{mean_dice:.4f}**（与参考标注{kind}）")
            lines.append("")
        lines.append("| 区域 | Dice | 提交体积 | 参考体积 | 体积差 | 漏分割 | 过度分割 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for region in (REGION_WT, REGION_TC, REGION_ET):
            r = evaluation["regions"].get(region, {})
            d = r.get("dice")
            lines.append(
                f"| {region} | {'—' if d is None else f'{d:.4f}'} "
                f"| {r.get('submitted_ml', 0):.2f} mL | {r.get('reference_ml', 0):.2f} mL "
                f"| {r.get('volume_delta_ml', 0):+.2f} mL "
                f"| {r.get('missed_voxels', 0):,} | {r.get('extra_voxels', 0):,} |"
            )
        lines.append("")
        for region in (REGION_WT, REGION_TC, REGION_ET):
            comment = evaluation["regions"].get(region, {}).get("comment")
            if comment:
                lines.append(f"- **{region}**：{comment}")
        lines.append("")
        if not evaluation.get("label_check_ok", True):
            lines.append(f"> **[提示]** 标签检查：{evaluation.get('label_check','')}")
            lines.append("")
    elif evaluation.get("reason"):
        lines.append(f"> **[提示]** {evaluation['reason']}")
        lines.append("")

    if text_review and text_review.get("checklist"):
        lines.append("### 文字答案核对")
        lines.append("")
        icons = {"hit": "[已覆盖]", "miss": "[待补充]", "info": "[提示]"}
        for item in text_review["checklist"]:
            lines.append(f"- {icons.get(item['status'],'•')} **{item['item']}** — {item['detail']}")
        lines.append("")
        if text_review.get("disclaimer"):
            lines.append(f"> {text_review['disclaimer']}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"> {CLINICAL_JUDGEMENT_NOTE}")
    return "\n".join(lines)
