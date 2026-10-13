"""Case registry: discovers cases on disk and declares how each is interpreted.

The whole application is organised around a single :class:`Case` object.  The
dashboard, viewer, report writer, teaching mode and research log all read the
same structure, so a case cannot drift out of sync between modules.

Every dataset gets an *adapter* that declares two things explicitly, because
getting either wrong produces plausible-looking but meaningless numbers:

1. **modality mapping** - which file/channel is T1, T1ce, T2, FLAIR.
2. **label convention** - how integer labels map to WT/TC/ET.

The declarations are checked against the real data at load time
(:func:`Case.verify`), and any mismatch is surfaced in the UI instead of
being silently absorbed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import metrics
from metrics import LabelConvention

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = RAW_DIR / "brats_subset"

# Data provenance tags shown in the UI.  The UI never lets these be ambiguous.
ORIGIN_PUBLIC = "public_dataset"
ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_UNKNOWN = "unknown"

ORIGIN_LABELS = {
    ORIGIN_PUBLIC: "公开数据集真实病例",
    ORIGIN_SYNTHETIC: "合成测试数据",
    ORIGIN_UNKNOWN: "来源未确认",
}


@dataclass
class Case:
    case_id: str
    dataset_source: str
    modalities: tuple[str, ...]
    image_paths: dict[str, str]
    label_convention: str
    data_origin: str = ORIGIN_UNKNOWN
    reference_mask_path: str | None = None
    prediction_mask_path: str | None = None
    corrected_mask_path: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    segmentation_status: str = "not_run"
    measurements: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    # Channel order the model expects, as canonical modality keys.
    model_channel_order: tuple[str, ...] = ("t1ce", "t1", "t2", "flair")

    @property
    def convention(self) -> LabelConvention:
        return metrics.CONVENTIONS[self.label_convention]

    @property
    def display_name(self) -> str:
        return self.case_id

    @property
    def origin_label(self) -> str:
        return ORIGIN_LABELS.get(self.data_origin, ORIGIN_LABELS[ORIGIN_UNKNOWN])

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_mask_path and Path(self.reference_mask_path).exists())

    def reference_path(self) -> Path | None:
        if not self.reference_mask_path:
            return None
        p = Path(self.reference_mask_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.exists() else None

    def prediction_path(self) -> Path | None:
        if not self.prediction_mask_path:
            return None
        p = Path(self.prediction_mask_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.exists() else None

    # -- integrity ------------------------------------------------------
    def verify(self, ref_labels: np.ndarray | None = None) -> list[str]:
        """Return a list of human-readable problems, empty if the case is sound."""
        problems: list[str] = []
        if self.label_convention not in metrics.CONVENTIONS:
            problems.append(f"未知的标签编码约定: {self.label_convention}")
        for mod, path in self.image_paths.items():
            p = Path(path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                problems.append(f"缺少影像文件: {mod} -> {path}")
        if ref_labels is not None:
            check = metrics.verify_convention(ref_labels, self.convention)
            if not check.ok:
                problems.extend(check.messages)
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_source": self.dataset_source,
            "modalities": list(self.modalities),
            "image_paths": self.image_paths,
            "reference_mask_path": self.reference_mask_path,
            "label_convention": self.label_convention,
            "data_origin": self.data_origin,
            "segmentation_status": self.segmentation_status,
            "measurements": self.measurements,
        }


# --------------------------------------------------------------------------
# Dataset adapters
# --------------------------------------------------------------------------
# BraTS-2020 ships one file per modality per patient.  The filename suffixes map
# onto canonical keys as follows; this mapping is asserted, not assumed.
BRATS2020_SUFFIX_TO_MODALITY = {
    "t1": "t1",
    "t1ce": "t1ce",
    "t2": "t2",
    "flair": "flair",
}

# The MONAI BraTS bundle expects 4-D input with channel order T1c, T1, T2, FLAIR
# (read from the bundle's configs/metadata.json -> network_data_format).
MONAI_BRATS_CHANNEL_ORDER = ("t1ce", "t1", "t2", "flair")


def _discover_brats2020(root: Path) -> list[Case]:
    """Find patients under ``data/raw/brats_subset`` that have all 4 modalities."""
    out: list[Case] = []
    if not root.exists():
        return out

    manifest_ids: list[str] | None = None
    manifest_file = root / "manifest.json"
    if manifest_file.exists():
        try:
            manifest_ids = [e["patient_id"] for e in json.loads(manifest_file.read_text())]
        except (json.JSONDecodeError, KeyError, TypeError):
            manifest_ids = None

    patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if manifest_ids:
        order = {pid: i for i, pid in enumerate(manifest_ids)}
        patient_dirs.sort(key=lambda p: order.get(p.name, 10_000))

    for pdir in patient_dirs:
        image_paths: dict[str, str] = {}
        for f in pdir.glob("*.nii*"):
            stem = f.name
            for ext in (".nii.gz", ".nii"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            suffix = stem.rsplit("_", 1)[-1].lower()
            if suffix in BRATS2020_SUFFIX_TO_MODALITY:
                image_paths[BRATS2020_SUFFIX_TO_MODALITY[suffix]] = str(f.relative_to(PROJECT_ROOT))
        seg = next((f for f in pdir.glob("*seg.nii*")), None)
        if set(image_paths) != set(BRATS2020_SUFFIX_TO_MODALITY):
            continue
        out.append(
            Case(
                case_id=pdir.name,
                dataset_source="BraTS 2020 (public, CC BY-NC 4.0)",
                modalities=tuple(sorted(image_paths)),
                image_paths=image_paths,
                reference_mask_path=str(seg.relative_to(PROJECT_ROOT)) if seg else None,
                label_convention=metrics.BRATS_2020.name,
                data_origin=ORIGIN_PUBLIC,
                model_channel_order=MONAI_BRATS_CHANNEL_ORDER,
                note="公开数据集真实病例，含专家参考标注。仅用于科研与教学演示。",
            )
        )
    return out


def _discover_synthetic(root: Path) -> list[Case]:
    """Synthetic phantom cases, generated by ``scripts/make_synthetic_case.py``.

    Always tagged :data:`ORIGIN_SYNTHETIC` so the UI can label them loudly.
    """
    out: list[Case] = []
    if not root.exists():
        return out
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_paths: dict[str, str] = {}
        for f in sdir.glob("*.nii*"):
            for mod in ("t1ce", "t1", "t2", "flair"):
                if f.name.startswith(f"{mod}.") or f.name.startswith(f"synth_{mod}"):
                    image_paths[mod] = str(f.relative_to(PROJECT_ROOT))
        seg = next((f for f in sdir.glob("*seg*.nii*")), None)
        if len(image_paths) != 4:
            continue
        out.append(
            Case(
                case_id=sdir.name,
                dataset_source="合成测试数据 (Synthetic phantom)",
                modalities=tuple(sorted(image_paths)),
                image_paths=image_paths,
                reference_mask_path=str(seg.relative_to(PROJECT_ROOT)) if seg else None,
                label_convention=metrics.BRATS_2020.name,
                data_origin=ORIGIN_SYNTHETIC,
                model_channel_order=MONAI_BRATS_CHANNEL_ORDER,
                note="程序生成的合成体模，仅用于验证界面与计算流程，不是真实病例。",
            )
        )
    return out


@dataclass
class CaseRegistry:
    cases: dict[str, Case] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, case: Case) -> None:
        self.cases[case.case_id] = case

    def get(self, case_id: str) -> Case | None:
        return self.cases.get(case_id)

    def ids(self) -> list[str]:
        return list(self.cases)

    def real_cases(self) -> list[Case]:
        return [c for c in self.cases.values() if c.data_origin == ORIGIN_PUBLIC]

    def with_reference(self) -> list[Case]:
        return [c for c in self.cases.values() if c.has_reference]

    def describe(self) -> str:
        real = len(self.real_cases())
        synth = sum(1 for c in self.cases.values() if c.data_origin == ORIGIN_SYNTHETIC)
        return f"{len(self.cases)} 例（真实公开数据 {real}，合成测试 {synth}）"


def load_registry(data_dir: Path | None = None) -> CaseRegistry:
    """Build the registry from whatever is available on disk.

    Real public cases always come first.  Synthetic phantoms exist only so the
    app remains demonstrable when the dataset has not been fetched yet.

    ``data_dir`` is threaded through explicitly rather than read from a module
    constant, so tests (and alternate deployments) can point the registry at
    another tree without monkey-patching module globals.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    reg = CaseRegistry()
    for case in _discover_brats2020(root / "raw" / "brats_subset"):
        reg.add(case)
    for case in _discover_synthetic(root / "synthetic"):
        reg.add(case)
    if not reg.cases:
        reg.warnings.append(
            "未发现任何病例数据。请运行 `python scripts/fetch_data.py --cases 6` "
            "下载公开 BraTS 数据，或运行 `python scripts/make_synthetic_case.py` "
            "生成明确标注的合成测试数据。"
        )
    return reg


def order_channels(case: Case, volumes: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-modality volumes into the channel order the model expects.

    This function is the single place where channel ordering happens.  It is
    deliberately explicit and is covered by tests, because silently feeding
    T2 where the model expects T1ce produces confident nonsense.
    """
    missing = [m for m in case.model_channel_order if m not in volumes]
    if missing:
        raise ValueError(f"缺少模型所需序列: {missing}")
    arrays = [np.asarray(volumes[m], dtype=np.float32) for m in case.model_channel_order]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"各序列空间尺寸不一致，无法堆叠: {sorted(shapes)}")
    return np.stack(arrays, axis=-1)


def channel_order_description(case: Case) -> str:
    labels = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}
    order = " → ".join(f"ch{i}:{labels[m]}" for i, m in enumerate(case.model_channel_order))
    return order
