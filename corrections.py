"""Human-corrected segmentation masks: import, validate, compare, version.

Why this module exists
----------------------
The research loop was missing its most important input.  Everything up to now
compared the *model* against the *dataset's* reference annotation; nothing
captured what a **person** decided the mask should be.  This module closes that
gap: a corrected mask can be imported (typically produced in 3D Slicer),
validated against the case, compared with the prediction, and stored as a new
immutable revision.

Three distinction rules are enforced here, not merely documented
----------------------------------------------------------------
1. **A user correction is not a dataset reference annotation.** They are stored
   in different fields, carry different provenance labels, and are never merged.
2. **An uploaded correction is not a clinician's sign-off.** Nothing in this
   module may be labelled as physician-reviewed; the provenance strings say the
   file was uploaded by a user, and say so verbatim in the UI and the report.
3. **Agreement between a prediction and a correction is not diagnostic
   accuracy.** It measures how far two masks differ, nothing more.

A demo path exists (re-importing a dataset reference annotation as if it were a
correction) so the flow can be shown without a radiologist on hand.  It is
labelled ``DEMO_REFERENCE_SIMULATION`` everywhere and can never be mistaken for
a real user edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import metrics
from metrics import REGION_ET, REGION_TC, REGION_WT

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
SOURCE_USER_UPLOAD = "user_upload"
SOURCE_DEMO_SIMULATION = "demo_reference_simulation"

SOURCE_LABELS = {
    SOURCE_USER_UPLOAD: "用户上传的修正掩膜",
    SOURCE_DEMO_SIMULATION: "演示数据（由参考标注模拟，非真实人工修正）",
}

#: Shown wherever a correction is displayed. Kept here so the wording cannot
#: drift between the UI, the exported CSV and the report.
CORRECTION_DISCLAIMER = (
    "人工修正掩膜由用户上传，**未经具备资质的医师审核**，"
    "不构成医生审核结果，也不代表临床诊断准确性。"
    "该结果仅用于科研与教学演示。"
)

DEMO_DISCLAIMER = (
    "⚠ 本次修正是**用数据集参考标注模拟的演示数据**，不是真实的人工修正操作，"
    "不得作为人工修正精度或医生审核结论使用。"
)

#: Upload formats the parser understands. Declared explicitly rather than
#: guessed, so a binary mask is never silently reinterpreted as a label map.
FORMAT_BRATS_LABELMAP = "brats_labelmap"
FORMAT_BINARY_REGION = "binary_region"

FORMAT_LABELS = {
    FORMAT_BRATS_LABELMAP: "BraTS 标签图（1=NCR/NET, 2=水肿, 4=强化）",
    FORMAT_BINARY_REGION: "单一区域二值掩膜（0/1，需指定区域）",
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class CorrectionValidation:
    """Result of checking an uploaded mask against the case it claims to belong to."""

    ok: bool = False
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    shape: tuple[int, ...] | None = None
    expected_shape: tuple[int, ...] | None = None
    affine_matches: bool = False
    orientation: str = ""
    expected_orientation: str = ""
    observed_labels: tuple[int, ...] = ()
    convention_ok: bool = False
    format: str = ""
    case_id_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "shape": list(self.shape) if self.shape else None,
            "expected_shape": list(self.expected_shape) if self.expected_shape else None,
            "affine_matches": self.affine_matches,
            "orientation": self.orientation,
            "expected_orientation": self.expected_orientation,
            "observed_labels": list(self.observed_labels),
            "convention_ok": self.convention_ok,
            "format": self.format,
            "case_id_checked": self.case_id_checked,
        }

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "掩膜校验通过：尺寸、空间方向与标签编码均与当前病例一致。"
        lines = []
        for problem in self.problems:
            lines.append(f"✗ {problem}")
        for warning in self.warnings:
            lines.append(f"· {warning}")
        return "　".join(lines)


def _aff2axcodes(affine: np.ndarray) -> str:
    import nibabel as nib

    return "".join(nib.aff2axcodes(np.asarray(affine, dtype=float)))


def parse_uploaded_mask(
    data: np.ndarray,
    upload_format: str,
    region: str | None,
    convention: metrics.LabelConvention,
) -> tuple[np.ndarray, list[str]]:
    """Convert an uploaded array into a canonical BraTS label map.

    The upload format is chosen by the user rather than inferred, because a
    binary mask and a label map containing only 0/1 are indistinguishable --
    guessing would silently turn "this voxel is tumour core" into "this voxel is
    NCR/NET" or vice versa.
    """
    arr = np.asarray(data)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"期望 3D 掩膜，实际为 {arr.ndim}D")

    warnings: list[str] = []
    arr = arr.astype(np.int16)

    # Normalise the common 0/1 float or bool encodings produced by editors.
    if arr.dtype.kind == "f":
        arr = np.rint(arr).astype(np.int16)

    if upload_format == FORMAT_BINARY_REGION:
        if region not in (REGION_WT, REGION_TC, REGION_ET):
            raise ValueError("二值掩膜必须指定所属区域（WT/TC/ET）")
        binary = arr > 0
        if int(binary.sum()) == 0:
            raise ValueError("上传的二值掩膜不含任何前景体素")
        label_map = metrics.masks_to_label_map(_region_to_channel_masks(binary, region))
        warnings.append(
            f"已按「{region} 区域二值掩膜」解析：外层区域被补齐为嵌套结构"
            f"（BraTS 标签图本身是互斥的，直接写入会丢失嵌套关系）")
        return label_map.astype(np.uint8), warnings

    if upload_format != FORMAT_BRATS_LABELMAP:
        raise ValueError(f"未知的上传格式: {upload_format}")

    check = metrics.verify_convention(arr.astype(np.uint8), convention)
    if not check.ok:
        raise ValueError(f"标签编码与当前病例不符：{check.summary()}")
    return arr.astype(np.uint8), warnings


def _region_to_channel_masks(binary: np.ndarray, region: str) -> dict[str, np.ndarray]:
    """Expand a single-region binary mask into the nested channel structure.

    A correction supplied as "this is the whole tumour" says nothing about the
    core or the enhancing part, so those are left empty.  A correction supplied
    as "this is the enhancing tumour" implies the voxels are also core and whole
    tumour, because those regions strictly contain it.
    """
    empty = np.zeros_like(binary, dtype=bool)
    if region == REGION_WT:
        return {REGION_WT: binary, REGION_TC: empty, REGION_ET: empty}
    if region == REGION_TC:
        return {REGION_WT: binary, REGION_TC: binary, REGION_ET: empty}
    return {REGION_WT: binary, REGION_TC: binary, REGION_ET: binary}


#: Case identifiers look like ``BraTS20_Training_001``.  Used to cross-check an
#: uploaded file's name against the case it is being attached to.
_CASE_ID_IN_NAME = re.compile(r"(BraTS\d{2,4}_[A-Za-z]+_\d{1,4})", re.IGNORECASE)


def check_case_identifier(source_name: str | None,
                          expected_case_id: str | None) -> tuple[bool, str]:
    """Cross-check an uploaded file's name against the case in view.

    Geometry checks alone cannot catch a mask belonging to a *different* case:
    every BraTS 2020 volume is 240x240x155 at 1 mm, so a mask for patient 002
    would satisfy shape, affine and orientation for patient 001 perfectly.
    The filename is the one place such a mix-up is usually visible.

    Returns ``(ok, message)``.  ``ok`` is False only on a positive mismatch --
    a name that carries no case identifier at all is not evidence of a problem,
    and is reported as a caveat instead.
    """
    if not expected_case_id:
        return True, ""
    found = _CASE_ID_IN_NAME.findall(str(source_name or ""))
    if not found:
        return True, (f"文件名中未包含病例编号，无法据文件名核对；"
                      f"已按当前病例 {expected_case_id} 归档，请自行确认掩膜来源")
    wrong = [f for f in found if f.lower() != expected_case_id.lower()]
    if wrong:
        return False, (f"文件名为其他病例（{', '.join(sorted(set(wrong)))}），"
                       f"当前病例是 {expected_case_id}；已拒绝导入")
    return True, ""


def validate_correction(
    label_map: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    affine: np.ndarray,
    expected_affine: np.ndarray,
    convention: metrics.LabelConvention,
    upload_format: str = "",
    affine_atol: float = 1e-3,
    source_name: str | None = None,
    expected_case_id: str | None = None,
) -> CorrectionValidation:
    """Check that an uploaded mask actually belongs to this case.

    Geometry is checked before anything else: a mask on the wrong grid would
    still produce Dice numbers, and those numbers would be meaningless.
    """
    result = CorrectionValidation(format=upload_format)
    arr = np.asarray(label_map)
    result.shape = tuple(arr.shape)
    result.expected_shape = tuple(expected_shape)

    if arr.shape != tuple(expected_shape):
        result.problems.append(
            f"掩膜尺寸 {tuple(arr.shape)} 与病例影像 {tuple(expected_shape)} 不一致，"
            f"无法叠加或比较")
    if arr.ndim == 3:
        result.orientation = _aff2axcodes(affine)
        result.expected_orientation = _aff2axcodes(expected_affine)
        if result.orientation != result.expected_orientation:
            result.problems.append(
                f"空间方向 {result.orientation} 与病例影像 "
                f"{result.expected_orientation} 不一致")

    a, b = np.asarray(affine, float), np.asarray(expected_affine, float)
    result.affine_matches = bool(a.shape == b.shape and np.allclose(a, b, atol=affine_atol))
    if not result.affine_matches:
        result.problems.append("affine 矩阵与病例影像不一致（体素间距或原点不同）")

    id_ok, id_message = check_case_identifier(source_name, expected_case_id)
    if id_message:
        result.case_id_checked = True
        (result.problems if not id_ok else result.warnings).append(id_message)

    result.observed_labels = tuple(int(v) for v in np.unique(arr))
    check = metrics.verify_convention(arr.astype(np.uint8), convention)
    result.convention_ok = check.ok
    if not check.ok:
        result.problems.append(f"标签编码不符合当前病例约定：{check.summary()}")

    if arr.ndim == 3 and not np.any(arr):
        result.problems.append("掩膜为空（不含任何前景体素）")

    # A correction identical to the prediction is legal but worth flagging.
    if arr.ndim == 3 and arr.size and not np.any(arr):
        result.warnings.append("空掩膜会使 Dice 为 0，请确认这是有意为之")

    result.ok = not result.problems
    return result


# --------------------------------------------------------------------------
# Comparison against the model prediction
# --------------------------------------------------------------------------
@dataclass
class CorrectionDiff:
    """How a corrected mask differs from the model prediction."""

    dice: dict[str, float | None] = field(default_factory=dict)
    predicted_ml: dict[str, float] = field(default_factory=dict)
    corrected_ml: dict[str, float] = field(default_factory=dict)
    volume_delta_ml: dict[str, float] = field(default_factory=dict)
    added_voxels: dict[str, int] = field(default_factory=dict)      # correction adds
    removed_voxels: dict[str, int] = field(default_factory=dict)    # correction removes
    mean_dice: float | None = None
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dice": self.dice,
            "predicted_ml": self.predicted_ml,
            "corrected_ml": self.corrected_ml,
            "volume_delta_ml": self.volume_delta_ml,
            "added_voxels": self.added_voxels,
            "removed_voxels": self.removed_voxels,
            "mean_dice": self.mean_dice,
            "spacing": list(self.spacing),
            "note": ("Dice 为模型预测与人工修正掩膜的一致性度量，"
                     "不代表临床诊断准确性。"),
        }


def compare_to_prediction(
    prediction: np.ndarray,
    correction: np.ndarray,
    spacing: tuple[float, float, float],
    convention: metrics.LabelConvention = metrics.BRATS_2020,
) -> CorrectionDiff:
    """Compare a correction with the model prediction, region by region."""
    pred = np.asarray(prediction)
    corr = np.asarray(correction)
    if pred.shape != corr.shape:
        raise ValueError(f"尺寸不一致：预测 {pred.shape} vs 修正 {corr.shape}")

    evaluation = metrics.evaluate_segmentation(corr, pred, spacing, convention)
    pred_masks = metrics.region_masks(pred, convention)
    corr_masks = metrics.region_masks(corr, convention)
    pred_vols = metrics.all_region_volumes(pred_masks, spacing)
    corr_vols = metrics.all_region_volumes(corr_masks, spacing)

    diff = CorrectionDiff(spacing=tuple(float(s) for s in spacing))
    for region in (REGION_WT, REGION_TC, REGION_ET):
        diff.dice[region] = evaluation.dice[region].dice
        diff.predicted_ml[region] = pred_vols[region].volume_ml
        diff.corrected_ml[region] = corr_vols[region].volume_ml
        diff.volume_delta_ml[region] = (corr_vols[region].volume_ml
                                        - pred_vols[region].volume_ml)
        diff.added_voxels[region] = int(np.logical_and(corr_masks[region],
                                                       ~pred_masks[region]).sum())
        diff.removed_voxels[region] = int(np.logical_and(pred_masks[region],
                                                         ~corr_masks[region]).sum())
    diff.mean_dice = evaluation.mean_dice()
    return diff


def diff_masks(prediction: np.ndarray, correction: np.ndarray,
               convention: metrics.LabelConvention = metrics.BRATS_2020) -> dict[str, np.ndarray]:
    """Per-region maps of what the correction adds and removes.

    ``added``   = the human marked it, the model did not.
    ``removed`` = the model marked it, the human did not.
    """
    pred_masks = metrics.region_masks(prediction, convention)
    corr_masks = metrics.region_masks(correction, convention)
    out: dict[str, np.ndarray] = {}
    for region in (REGION_WT, REGION_TC, REGION_ET):
        out[f"{region}_added"] = np.logical_and(corr_masks[region], ~pred_masks[region])
        out[f"{region}_removed"] = np.logical_and(pred_masks[region], ~corr_masks[region])
    return out


# --------------------------------------------------------------------------
# Versioned storage
# --------------------------------------------------------------------------
@dataclass
class CorrectionRecord:
    """A stored correction revision. Immutable once written."""

    correction_id: str
    case_id: str
    revision: int
    path: str
    source: str
    source_label: str
    created_at: str
    validation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_demo(self) -> bool:
        return self.source == SOURCE_DEMO_SIMULATION


def correction_dir(root: Path, case_id: str) -> Path:
    """Corrections live under their own directory, never beside predictions."""
    return root / "outputs" / "corrections" / case_id


def save_correction(
    label_map: np.ndarray,
    affine: np.ndarray,
    *,
    root: Path,
    case_id: str,
    revision: int,
    source: str = SOURCE_USER_UPLOAD,
    validation: dict[str, Any] | None = None,
    metrics_payload: dict[str, Any] | None = None,
    notes: str = "",
    store: Any = None,
) -> CorrectionRecord:
    """Write a correction as a new immutable revision.

    The prediction file is never touched: corrections are stored under
    ``outputs/corrections/<case>/`` with a monotonically increasing revision, so
    the original model output remains available for comparison forever.
    """
    import nibabel as nib

    folder = correction_dir(root, case_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{case_id}_corrected_r{revision}.nii.gz"
    if path.exists():
        raise FileExistsError(
            f"修订 {revision} 已存在（{path.name}）；修正记录不可覆盖")

    nib.save(nib.Nifti1Image(np.asarray(label_map).astype(np.uint8), np.asarray(affine)),
             str(path))
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if store is not None:
        store.record_correction(
            case_id, revision, str(path), source,
            validation=validation or {}, metrics_payload=metrics_payload or {},
            notes=notes)
        store.upsert_case(case_id, corrected_mask_path=str(path))

    return CorrectionRecord(
        correction_id=f"{case_id}_r{revision}", case_id=case_id, revision=revision,
        path=str(path), source=source, source_label=SOURCE_LABELS.get(source, source),
        created_at=created, validation=validation or {},
        metrics=metrics_payload or {}, notes=notes)


def load_correction(path: str | Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


# --------------------------------------------------------------------------
# Feedback integration
# --------------------------------------------------------------------------
def suggest_error_type_from_diff(diff: CorrectionDiff) -> str:
    """Pre-select a research error category from the correction.

    Only a suggestion: the reviewer confirms or overrides before anything is
    written to the research record.
    """
    total_added = sum(diff.added_voxels.values())
    total_removed = sum(diff.removed_voxels.values())
    if total_added == 0 and total_removed == 0:
        return "other"
    if total_removed > 4 * max(total_added, 1):
        return "over_segmentation"
    if total_added > 4 * max(total_removed, 1):
        return "under_segmentation"
    return "boundary_deviation"


def feedback_note(diff: CorrectionDiff, source: str, revision: int) -> str:
    """Default note text stored with the research feedback record.

    States the provenance explicitly so a future reader cannot mistake a demo
    simulation for a real correction.
    """
    label = SOURCE_LABELS.get(source, source)
    parts = [f"第 {revision} 版修正（{label}）"]
    if source == SOURCE_DEMO_SIMULATION:
        parts.append(DEMO_DISCLAIMER)
    for region in (REGION_WT, REGION_TC, REGION_ET):
        delta = diff.volume_delta_ml.get(region, 0.0)
        parts.append(f"{region} 体积变化 {delta:+.2f} mL"
                     f"（+{diff.added_voxels.get(region, 0)} / "
                     f"-{diff.removed_voxels.get(region, 0)} 体素）")
    return "；".join(parts)
