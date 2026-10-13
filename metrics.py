"""Deterministic segmentation metrics: region decoding, volumes and Dice.

Everything in this module is closed-form arithmetic on integer label maps.
No model, no LLM and no randomness is involved: given the same mask and the
same voxel spacing, the numbers are reproducible bit-for-bit.  This matters
because the report and the research dashboard quote these numbers as facts.

Region definitions
------------------
Brain-tumour challenge data uses *nested* sub-regions rather than mutually
exclusive classes:

    ET  (enhancing tumour)                ⊂ TC ⊂ WT
    TC  (tumour core) = NCR/NET + ET
    WT  (whole tumour) = NCR/NET + ED + ET

Because the regions overlap, we never treat the three model output channels
as three independent labels.  We always convert to a BraTS-style integer
label map first and then derive the three binary masks from it, which makes
the nesting structurally impossible to get wrong.

Label encodings differ between datasets, so the encoding is *declared per
dataset* (see :class:`LabelConvention`) and then *verified against the actual
data* by :func:`verify_convention`.  We never silently guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

# Canonical region keys used across the whole application.
REGION_WT = "WT"
REGION_TC = "TC"
REGION_ET = "ET"
REGIONS: tuple[str, str, str] = (REGION_WT, REGION_TC, REGION_ET)

REGION_LONG_NAMES = {
    REGION_WT: "Whole Tumour",
    REGION_TC: "Tumour Core",
    REGION_ET: "Enhancing Tumour",
}
REGION_NAMES_ZH = {
    REGION_WT: "全肿瘤区",
    REGION_TC: "肿瘤核心区",
    REGION_ET: "强化肿瘤区",
}


@dataclass(frozen=True)
class LabelConvention:
    """How integer label values map onto the three nested BraTS regions."""

    name: str
    wt_labels: tuple[int, ...]
    tc_labels: tuple[int, ...]
    et_labels: tuple[int, ...]
    source: str

    @property
    def all_labels(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.wt_labels)))

    def describe(self) -> str:
        return (
            f"{self.name}: WT={list(self.wt_labels)} "
            f"TC={list(self.tc_labels)} ET={list(self.et_labels)}"
        )


# BraTS 2018-2021 ground truth: 1 = NCR/NET, 2 = oedema, 4 = enhancing tumour.
# Note there is deliberately no label 3 in this encoding.
BRATS_2020 = LabelConvention(
    name="brats_2020",
    wt_labels=(1, 2, 4),
    tc_labels=(1, 4),
    et_labels=(4,),
    source="BraTS 2018-2021 ground truth (1=NCR/NET, 2=ED, 4=ET)",
)

# Some early/derived BraTS releases used 3 for enhancing tumour instead of 4.
BRATS_LEGACY = LabelConvention(
    name="brats_legacy",
    wt_labels=(1, 2, 3),
    tc_labels=(1, 3),
    et_labels=(3,),
    source="BraTS-derived releases that encode enhancing tumour as 3",
)

# Medical Segmentation Decathlon Task01_BrainTumour:
# 1 = oedema, 2 = non-enhancing tumour, 3 = enhancing tumour.
MSD_BRAIN_TUMOUR = LabelConvention(
    name="msd_brain_tumour",
    wt_labels=(1, 2, 3),
    tc_labels=(2, 3),
    et_labels=(3,),
    source="MSD Task01_BrainTumour (1=ED, 2=NET, 3=ET)",
)

CONVENTIONS: dict[str, LabelConvention] = {
    c.name: c for c in (BRATS_2020, BRATS_LEGACY, MSD_BRAIN_TUMOUR)
}


# --------------------------------------------------------------------------
# Convention verification
# --------------------------------------------------------------------------
@dataclass
class ConventionCheck:
    """Outcome of checking a declared convention against real label values."""

    convention: str
    observed_labels: tuple[int, ...]
    ok: bool
    messages: list[str]

    def summary(self) -> str:
        head = "OK" if self.ok else "MISMATCH"
        return f"[{head}] {self.convention} observed={list(self.observed_labels)}; " + "; ".join(self.messages)


def verify_convention(label_map: np.ndarray, convention: LabelConvention) -> ConventionCheck:
    """Check that a label map actually contains the labels the convention claims.

    This is the guard rail that stops us from computing Dice against the wrong
    region definition.  It reports rather than raises, so the UI can surface a
    precise warning instead of a stack trace.
    """
    observed = tuple(int(v) for v in np.unique(label_map))
    messages: list[str] = []
    ok = True

    expected = set(convention.all_labels) | {0}
    unexpected = sorted(set(observed) - expected)
    if unexpected:
        ok = False
        messages.append(
            f"labels {unexpected} are present in the data but not described by "
            f"'{convention.name}' - region definitions may be wrong"
        )

    missing = sorted(set(convention.all_labels) - set(observed))
    if missing:
        # A missing region is not necessarily an error: 29% of BraTS cases have
        # no enhancing tumour at all, so label 4 legitimately disappears.
        messages.append(
            f"labels {missing} absent from this case (normal when a region is "
            f"genuinely not present)"
        )

    if not messages:
        messages.append("all declared labels observed")
    return ConventionCheck(convention.name, observed, ok, messages)


def suggest_convention(label_map: np.ndarray) -> LabelConvention | None:
    """Best-effort guess at the encoding, used only for *diagnostics*.

    The application never relies on this to pick a convention; the dataset
    adapter declares it.  This exists so that an unexpected dataset produces a
    helpful message rather than silently wrong numbers.
    """
    observed = {int(v) for v in np.unique(label_map)}
    if 4 in observed:
        return BRATS_2020
    if 3 in observed and 2 in observed and 1 in observed:
        # Ambiguous between legacy-BraTS and MSD; prefer MSD's ordering only
        # when the caller has told us the source is MSD.
        return BRATS_LEGACY
    return None


# --------------------------------------------------------------------------
# Label map -> regional masks
# --------------------------------------------------------------------------
def region_masks(label_map: np.ndarray, convention: LabelConvention) -> dict[str, np.ndarray]:
    """Return the three nested binary masks for a label map.

    The masks are derived from a single label map, so ``ET ⊆ TC ⊆ WT`` holds
    by construction.
    """
    lm = np.asarray(label_map)
    return {
        REGION_WT: np.isin(lm, convention.wt_labels),
        REGION_TC: np.isin(lm, convention.tc_labels),
        REGION_ET: np.isin(lm, convention.et_labels),
    }


def masks_to_label_map(
    channel_masks: Mapping[str, np.ndarray],
    *,
    et_value: int = 4,
    tc_value: int = 1,
    wt_value: int = 2,
) -> np.ndarray:
    """Collapse three possibly-overlapping region masks into one label map.

    Precedence is ET > TC > WT, exactly as the MONAI BraTS bundle does in its
    own post-processing.  This is what makes nested predictions safe to store
    as a single integer volume.
    """
    shape = next(iter(channel_masks.values())).shape
    out = np.zeros(shape, dtype=np.uint8)
    wt = channel_masks.get(REGION_WT, np.zeros(shape, bool))
    tc = channel_masks.get(REGION_TC, np.zeros(shape, bool))
    et = channel_masks.get(REGION_ET, np.zeros(shape, bool))
    out[wt] = wt_value
    out[tc] = tc_value
    out[et] = et_value
    return out


# --------------------------------------------------------------------------
# Volumes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeResult:
    region: str
    voxel_count: int
    volume_ml: float
    voxel_volume_ml: float


def voxel_volume_ml(spacing: Iterable[float]) -> float:
    """Volume of one voxel in millilitres from voxel spacing in millimetres."""
    sx, sy, sz = (float(s) for s in spacing)
    return (sx * sy * sz) / 1000.0


def region_volume(mask: np.ndarray, spacing: Iterable[float], region: str) -> VolumeResult:
    vox_ml = voxel_volume_ml(spacing)
    count = int(np.count_nonzero(mask))
    return VolumeResult(region, count, count * vox_ml, vox_ml)


def all_region_volumes(
    masks: Mapping[str, np.ndarray], spacing: Iterable[float]
) -> dict[str, VolumeResult]:
    return {r: region_volume(masks[r], spacing, r) for r in REGIONS if r in masks}


# --------------------------------------------------------------------------
# Dice
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DiceResult:
    region: str
    dice: float | None
    intersection: int
    pred_voxels: int
    ref_voxels: int
    union: int
    note: str = ""

    @property
    def defined(self) -> bool:
        return self.dice is not None


def dice(pred: np.ndarray, ref: np.ndarray, region: str = "") -> DiceResult:
    """Sørensen-Dice coefficient between two binary masks.

    Edge cases are handled explicitly rather than silently:

    * both empty -> the coefficient is mathematically 0/0.  Standard practice in
      the BraTS challenge is to score a correct empty prediction as 1.0, so we
      return 1.0 and flag it in ``note`` so it can be excluded from aggregates
      if desired.
    * exactly one empty -> 0.0.
    """
    p = np.asarray(pred, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: prediction {p.shape} vs reference {r.shape}")

    np_count, nr_count = int(p.sum()), int(r.sum())
    inter = int(np.logical_and(p, r).sum())
    union = np_count + nr_count - inter

    if np_count + nr_count == 0:
        return DiceResult(region, 1.0, 0, 0, 0, 0, note="both masks empty; scored as 1.0 by convention")
    d = (2.0 * inter) / (np_count + nr_count)
    note = ""
    if np_count == 0 or nr_count == 0:
        note = "one mask empty; Dice is 0 by definition"
    return DiceResult(region, d, inter, np_count, nr_count, union, note)


@dataclass
class SegmentationEvaluation:
    """Per-region Dice plus the voxel-level error breakdown."""

    dice: dict[str, DiceResult]
    missed: dict[str, int]      # reference voxels the prediction did not find
    extra: dict[str, int]       # predicted voxels with no reference support
    spacing: tuple[float, float, float]

    def mean_dice(self) -> float | None:
        vals = [d.dice for d in self.dice.values() if d.dice is not None]
        return float(np.mean(vals)) if vals else None

    def as_dict(self) -> dict:
        return {
            "dice": {k: v.dice for k, v in self.dice.items()},
            "missed_voxels": self.missed,
            "extra_voxels": self.extra,
        }


def evaluate_segmentation(
    pred_label_map: np.ndarray,
    ref_label_map: np.ndarray,
    spacing: Iterable[float],
    convention: LabelConvention = BRATS_2020,
) -> SegmentationEvaluation:
    """Compare a predicted label map against a reference label map.

    Both inputs must already be in the *same* label encoding and the same
    voxel grid.  Callers are responsible for resampling; this function refuses
    to compare arrays of different shape rather than guessing.
    """
    pred_label_map = np.asarray(pred_label_map)
    ref_label_map = np.asarray(ref_label_map)
    if pred_label_map.shape != ref_label_map.shape:
        raise ValueError(
            f"prediction shape {pred_label_map.shape} != reference shape "
            f"{ref_label_map.shape}; resample before evaluating"
        )

    pm = region_masks(pred_label_map, convention)
    rm = region_masks(ref_label_map, convention)

    dice_out, missed, extra = {}, {}, {}
    for region in REGIONS:
        d = dice(pm[region], rm[region], region)
        dice_out[region] = d
        missed[region] = int(np.logical_and(rm[region], ~pm[region]).sum())
        extra[region] = int(np.logical_and(pm[region], ~rm[region]).sum())

    sp = tuple(float(s) for s in spacing)
    return SegmentationEvaluation(dice_out, missed, extra, sp)  # type: ignore[arg-type]


def difference_maps(pred_label_map: np.ndarray, ref_label_map: np.ndarray,
                    convention: LabelConvention = BRATS_2020) -> dict[str, np.ndarray]:
    """Per-region false-negative / false-positive masks, for error visualisation."""
    pm = region_masks(pred_label_map, convention)
    rm = region_masks(ref_label_map, convention)
    out = {}
    for region in REGIONS:
        out[f"{region}_FN"] = np.logical_and(rm[region], ~pm[region])
        out[f"{region}_FP"] = np.logical_and(pm[region], ~rm[region])
    return out


def error_profile(missed: Mapping[str, int], extra: Mapping[str, int]) -> str:
    """Human-readable error classification hint derived from voxel counts.

    This only *suggests* a label for the reviewer; the reviewer always makes
    the final call.  A model is never allowed to write its own error category
    into the research record unattended.
    """
    total_missed = sum(missed.get(r, 0) for r in REGIONS)
    total_extra = sum(extra.get(r, 0) for r in REGIONS)
    if total_missed == 0 and total_extra == 0:
        return "无差异"
    if total_extra > 3 * max(total_missed, 1):
        return "疑似过度分割"
    if total_missed > 3 * max(total_extra, 1):
        return "疑似漏分割"
    return "疑似边界偏差"
