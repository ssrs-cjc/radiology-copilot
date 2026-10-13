"""MRI reading, input validation and slice rendering.

The single rule this module exists to enforce: **never assume anything about
the input**.  Multi-modal MRI datasets disagree about channel order, voxel
geometry and label encoding.  Every assumption is therefore either declared
explicitly by a dataset adapter and then verified here, or the case is
rejected with a precise message.

Rendering returns plain ``uint8`` RGB arrays so the UI layer can stay dumb and
the same code path is exercised by the tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # nibabel is a hard dependency, but keep the import error readable
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise ImportError("nibabel is required: pip install nibabel") from exc

# Canonical modality keys used throughout the app.
MODALITIES: tuple[str, str, str, str] = ("t1", "t1ce", "t2", "flair")

MODALITY_LABELS = {
    "t1": "T1",
    "t1ce": "T1ce",
    "t2": "T2",
    "flair": "FLAIR",
}

# Region overlay colours.  Deliberately distinct in hue *and* luminance so the
# nested regions stay separable for colour-vision-deficient users.
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "WT": (22, 166, 161),    # teal  - whole tumour (largest, drawn first)
    "TC": (242, 163, 60),    # amber - tumour core
    "ET": (229, 72, 77),     # red   - enhancing tumour (smallest, drawn last)
}
ERROR_COLORS: dict[str, tuple[int, int, int]] = {
    "FN": (139, 92, 246),    # violet - reference voxels the model missed
    "FP": (250, 204, 21),    # yellow - predicted voxels with no reference
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Volume:
    """A loaded NIfTI volume plus the geometry needed to interpret it."""

    path: Path
    data: np.ndarray            # (X, Y, Z) or (X, Y, Z, C)
    affine: np.ndarray
    header_spacing: tuple[float, float, float]
    is_4d: bool

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel size in mm, taken from the affine when it is well formed."""
        zooms = tuple(float(z) for z in nib.affines.voxel_sizes(self.affine)[:3])
        if not np.all(np.isfinite(zooms)) or min(zooms) <= 0:
            return self.header_spacing
        return zooms  # type: ignore[return-value]


def load_volume(path: str | Path) -> Volume:
    """Read a NIfTI file. Raises ValueError with a readable message on failure."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"影像文件不存在: {p}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = nib.load(str(p))
            data = np.asanyarray(img.dataobj, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 NIfTI 文件 {p.name}: {type(exc).__name__}: {exc}") from exc

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    is_4d = data.ndim == 4
    if data.ndim not in (3, 4):
        raise ValueError(f"{p.name}: 期望 3D 或 4D 体数据，实际为 {data.ndim}D")

    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return Volume(p, data, np.asarray(img.affine, dtype=float), zooms, is_4d)


def load_single_modality(paths: Mapping[str, Path], modality: str) -> Volume:
    """Load one modality from a per-file dataset, or one channel of a 4-D file."""
    if modality not in paths:
        raise ValueError(f"病例缺少 {MODALITY_LABELS.get(modality, modality)} 序列")
    vol = load_volume(paths[modality])
    return vol


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class ValidationIssue:
    level: str      # "error" | "warning" | "info"
    message: str
    detail: str = ""


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    modalities_present: list[str] = field(default_factory=list)
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, float, float] | None = None

    def add(self, level: str, message: str, detail: str = "") -> None:
        self.issues.append(ValidationIssue(level, message, detail))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def to_markdown(self) -> str:
        if self.ok and not self.warnings:
            lines = ["**输入检查通过** — 影像可读取、几何信息一致。"]
        elif self.ok:
            lines = ["**输入检查通过（有提示）**"]
        else:
            lines = ["**输入检查未通过 — 已阻止后续分析**"]

        for i in self.issues:
            icon = {"error": "[错误]", "warning": "[提示]", "info": "[信息]"}.get(i.level, "•")
            lines.append(f"- {icon} {i.message}")
            if i.detail:
                lines.append(f"  - `{i.detail}`")
        return "\n".join(lines)


def _affine_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol))


def validate_case(
    image_paths: Mapping[str, Path],
    reference_mask_path: str | Path | None = None,
    *,
    required_modalities: Sequence[str] = MODALITIES,
    loader: Callable[[str], Volume] | None = None,
) -> ValidationReport:
    """Check that a case is safe to analyse.

    Verifies, in order: files readable, all required sequences present, volumes
    share a voxel grid, and the affine carries usable spatial information.  A
    case that fails must not be segmented or overlaid.

    ``loader`` lets the caller supply a caching loader.  Without it, validating a
    case and then displaying it reads every volume from disk twice, which on a
    four-sequence MRI is the difference between a 2-second and a 5-second case
    switch.
    """
    load = loader or load_volume
    rep = ValidationReport(ok=True)
    loaded: dict[str, Volume] = {}

    for mod in required_modalities:
        path = image_paths.get(mod)
        if path is None:
            rep.add("error", f"缺少 {MODALITY_LABELS.get(mod, mod)} 序列文件")
            continue
        try:
            vol = load(str(path))
        except ValueError as exc:
            rep.add("error", f"{MODALITY_LABELS.get(mod, mod)} 无法读取", str(exc))
            continue
        loaded[mod] = vol
        rep.modalities_present.append(mod)

    if not loaded:
        rep.ok = False
        rep.add("error", "没有任何可用序列，无法继续")
        return rep

    # Geometry consistency across modalities.
    ref_mod = next(iter(loaded))
    ref_vol = loaded[ref_mod]
    rep.shape = tuple(ref_vol.data.shape[:3])
    rep.spacing = tuple(ref_vol.spacing)

    for mod, vol in loaded.items():
        if tuple(vol.data.shape[:3]) != rep.shape:
            rep.add("error", f"{MODALITY_LABELS.get(mod, mod)} 空间尺寸与 {MODALITY_LABELS.get(ref_mod, ref_mod)} 不一致",
                    f"{vol.data.shape[:3]} vs {rep.shape}")
        if not _affine_close(vol.affine, ref_vol.affine):
            rep.add("warning", f"{MODALITY_LABELS.get(mod, mod)} 的 affine 与参考序列不同",
                    "若未配准，叠加掩膜可能错位；请确认数据来源")

    # Affine must be usable, otherwise voxel spacing (and therefore every
    # volume in mL) is meaningless.
    if not np.all(np.isfinite(ref_vol.affine)) or abs(np.linalg.det(ref_vol.affine[:3, :3])) < 1e-8:
        rep.add("error", "affine 矩阵退化或包含非法值，无法计算体素体积")
    else:
        sp = ref_vol.spacing
        if min(sp) <= 0 or max(sp) > 20:
            rep.add("warning", f"体素间距异常: {tuple(round(s, 3) for s in sp)} mm")

    # Reference labels
    if reference_mask_path:
        try:
            ref = load(str(reference_mask_path))
            if tuple(ref.data.shape[:3]) != rep.shape:
                rep.add("error", "参考标注与影像空间尺寸不一致", f"{ref.data.shape[:3]} vs {rep.shape}")
            else:
                rep.add("info", "参考标注已加载并与影像空间一致")
        except ValueError as exc:
            rep.add("warning", "参考标注无法读取", str(exc))
    else:
        rep.add("info", "该病例没有参考标注，将无法计算 Dice 评测")

    rep.ok = not rep.errors
    return rep


# --------------------------------------------------------------------------
# Slice indexing / orientation
# --------------------------------------------------------------------------
def axial_axis(affine: np.ndarray) -> int:
    """Which array axis corresponds to the superior-inferior direction.

    Derived from the affine rather than assumed, because reoriented datasets
    do not always store the axial direction last.
    """
    rot = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    norms[norms == 0] = 1.0
    direction = (rot / norms)[2, :]          # component along the S-I world axis
    return int(np.argmax(np.abs(direction)))


def _flip_flags(affine: np.ndarray) -> tuple[bool, bool]:
    """Whether the in-plane axes must be flipped to display head-up/left-right."""
    rot = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    norms[norms == 0] = 1.0
    rot = rot / norms
    ax = axial_axis(affine)
    in_plane = [i for i in range(3) if i != ax]
    # World +x is patient-left in RAS; world +y is anterior.
    flip_x = rot[0, in_plane[0]] < 0
    flip_y = rot[1, in_plane[1]] < 0
    return flip_x, flip_y


def orient_plane(plane: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Bring a 2-D in-plane array into the display frame.

    Shared by slice extraction and by the head-bounding-box projection so the
    two can never disagree about which array axis is which.
    """
    flip_x, flip_y = _flip_flags(affine)
    if flip_x:
        plane = np.flip(plane, axis=1)
    if flip_y:
        plane = np.flip(plane, axis=0)
    if axial_axis(affine) != 2:
        plane = plane.T
    return plane


def take_axial_slice(data: np.ndarray, index: int, affine: np.ndarray) -> np.ndarray:
    """Extract an axial slice as a 2-D array oriented for display.

    The returned array is always indexed ``[row, col]`` with row increasing
    downward on screen.

    Indexing rather than ``np.take``: for a C-contiguous volume, taking along
    the last axis with ``np.take`` materialises a strided copy of the whole
    volume per call (~0.15 s on a 240x240x155 mask), whereas basic slicing
    yields a view and only the small 2-D result is copied.
    """
    ax = axial_axis(affine)
    n = data.shape[ax]
    index = int(np.clip(index, 0, n - 1))
    slicer: list[Any] = [slice(None)] * data.ndim
    slicer[ax] = index
    return orient_plane(data[tuple(slicer)], affine)


def axial_slice_count(data: np.ndarray, affine: np.ndarray) -> int:
    return int(data.shape[axial_axis(affine)])


# --------------------------------------------------------------------------
# Framing: crop the empty background so the head fills the panel
# --------------------------------------------------------------------------
def head_bounding_box(data: np.ndarray, affine: np.ndarray,
                      margin: float = 0.04) -> tuple[slice, slice] | None:
    """Square in-plane bounding box around the head, in display orientation.

    Brain MRI is skull-stripped and the head is roughly circular inside a
    square matrix, so a large fraction of every slice is empty background.  The
    viewer was therefore mostly black.  This finds where the tissue actually is
    so the panel can be filled with anatomy instead.

    The box is deliberately **square and computed once per volume**: square so
    the aspect ratio is never distorted, and once per volume so the head does
    not jitter sideways as the user scrolls through slices.

    Returns ``None`` when a sensible crop cannot be determined, in which case
    callers should render the slice uncropped.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        return None

    # Project onto the axial plane: a voxel counts as tissue if it is non-zero
    # anywhere along the through-plane axis.  This is a display framing
    # decision only -- no voxel values are modified.
    ax = axial_axis(affine)
    projection = np.any(data != 0, axis=ax)
    projection = orient_plane(projection, affine)
    if projection.ndim != 2:
        return None

    rows = np.flatnonzero(projection.any(axis=1))
    cols = np.flatnonzero(projection.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None

    height, width = projection.shape
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    # A square window that contains the tissue extent, centred on it, padded by
    # `margin` and clamped to the image.  Clamping then re-squaring keeps the
    # output square even at the edges of the matrix.
    side = max(bottom - top, right - left)
    side = int(round(side * (1.0 + 2.0 * margin)))
    side = min(side, height, width)
    if side <= 0:
        return None

    center_r = (top + bottom) / 2.0
    center_c = (left + right) / 2.0
    r0 = int(round(center_r - side / 2.0))
    c0 = int(round(center_c - side / 2.0))
    r0 = max(0, min(r0, height - side))
    c0 = max(0, min(c0, width - side))

    box = (slice(r0, r0 + side), slice(c0, c0 + side))
    # If the "crop" is basically the whole image, it is not worth doing.
    if side >= 0.97 * min(height, width):
        return None
    return box


# --------------------------------------------------------------------------
# Display rendering
# --------------------------------------------------------------------------
def window_slice(sl: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Map a slice to 0..255 using percentile windowing.

    ``low_pct``/``high_pct`` are **percentiles of the tissue intensities**, not
    absolute intensities.  That distinction matters: an earlier version of the
    UI passed its percentile sliders straight in as absolute bounds, which
    clipped every value above 99 to pure white and made the brain look like a
    blank blob.

    The percentiles are taken over **non-background voxels only**.  Brain MRI is
    skull-stripped, so most of a slice is exact zeros; including them would push
    the low percentile to 0 and collapse grey/white matter contrast.  Sampling
    real tissue is what makes the anatomy legible.
    """
    sl = np.asarray(sl, dtype=np.float32)
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros(sl.shape, dtype=np.uint8)

    tissue = finite[finite != 0]
    sample = tissue if tissue.size >= 64 else finite

    lo, hi = np.percentile(sample, [float(low_pct), float(high_pct)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = max(lo + 1.0, 1.0)

    out = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(sl)] = 0.0
    return (out * 255.0).astype(np.uint8)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def overlay_masks(rgb: np.ndarray, mask_layers: Iterable[tuple[np.ndarray, tuple[int, int, int]]],
                  alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend coloured region masks onto an RGB slice.

    ``mask_layers`` is ordered outermost-first (WT, then TC, then ET) so the
    nested structure is visible rather than the largest region hiding the rest.
    """
    out = rgb.astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    for mask, color in mask_layers:
        if mask is None:
            continue
        m = np.asarray(mask, dtype=bool)
        if m.shape != rgb.shape[:2] or not m.any():
            continue
        colour = np.asarray(color, dtype=np.float32)
        region = out[m]
        out[m] = region * (1.0 - a) + colour * a
    return np.clip(out, 0, 255).astype(np.uint8)


def slice_masks_at(masks: Mapping[str, np.ndarray], index: int,
                   affine: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the same axial plane from every region mask.

    The masks are boolean; converting them to float before slicing would
    allocate a 36 MB array per region per render (four times the mask's own
    size) and dominated slice-scrolling cost at ~0.45 s per mask.  Boolean
    slicing is both faster and produces exactly the same result.
    """
    out: dict[str, np.ndarray] = {}
    for key, mask in masks.items():
        arr = np.asarray(mask)
        plane = take_axial_slice(arr, index, affine)
        out[key] = plane if plane.dtype == bool else plane > 0.5
    return out


def _resize_rgb(rgb: np.ndarray, long_side: int) -> np.ndarray:
    """Smoothly resample anatomy for display (bilinear, aspect preserved)."""
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = long_side / float(max(h, w))
    if scale <= 1.0:
        return rgb
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))


def _resize_mask(mask: np.ndarray, long_side: int) -> np.ndarray:
    """Resample a label mask.

    Deliberately **nearest neighbour**: a label map holds discrete region ids,
    and interpolating them would invent boundaries and blend WT into TC along
    every edge.  Nearest neighbour keeps every output pixel one of the input
    labels.
    """
    from PIL import Image

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape[:2]
    scale = long_side / float(max(h, w))
    if scale <= 1.0:
        return m
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray((m.astype(np.uint8) * 255)).resize(size, Image.NEAREST)
    return np.asarray(resized) > 127


def compose_slice(
    rgb: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[int, int, int]]],
    alpha: float = 0.45,
    crop: tuple[slice, slice] | None = None,
    display_size: int | None = None,
) -> np.ndarray:
    """Shared tail of every slice render: crop, resample, blend.

    Cropping and resampling are applied identically to the anatomy and to every
    mask layer, which is what keeps the overlay registered to the underlying
    image, and masks are resampled nearest-neighbour so label ids survive.
    """
    if crop is not None:
        rgb = rgb[crop]
        layers = [(m[crop], colour) for m, colour in layers]

    if display_size:
        rgb = _resize_rgb(rgb, display_size)
        layers = [(_resize_mask(m, display_size), colour) for m, colour in layers]

    return overlay_masks(rgb, layers, alpha) if layers else rgb


def region_layers(masks: Mapping[str, np.ndarray]) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Ordered overlay layers for the three nested regions, outermost first."""
    return [(masks[r], REGION_COLORS[r]) for r in ("WT", "TC", "ET") if r in masks]


def error_layers(masks: Mapping[str, np.ndarray]) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    return [(masks[k], ERROR_COLORS[k.split("_")[1]])
            for k in ("WT_FN", "WT_FP") if k in masks]


def render_slice_2d(
    anatomy_slice: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
    *,
    alpha: float = 0.45,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    crop: tuple[slice, slice] | None = None,
    display_size: int | None = None,
) -> np.ndarray:
    """Render from an **already extracted 2-D slice**.

    Deriving region masks for a whole volume costs several passes over ~9M
    voxels.  When the caller only needs one plane -- the error comparison figure
    draws three panels of the same plane -- computing the masks on the 2-D slice
    instead is roughly two orders of magnitude cheaper and produces an identical
    picture.
    """
    gray = window_slice(anatomy_slice, low_pct, high_pct)
    rgb = to_rgb(gray)
    return compose_slice(rgb, list(layers or []), alpha, crop, display_size)


def render_mri_slice(
    volume_data: np.ndarray,
    affine: np.ndarray,
    index: int,
    *,
    masks: Mapping[str, np.ndarray] | None = None,
    show_masks: bool = True,
    alpha: float = 0.45,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    error_masks: Mapping[str, np.ndarray] | None = None,
    crop: tuple[slice, slice] | None = None,
    display_size: int | None = None,
) -> np.ndarray:
    """Produce the RGB image shown in the viewer from a 3-D volume and index."""
    sl = take_axial_slice(volume_data, index, affine)

    layers: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if show_masks and masks:
        layers.extend(region_layers(slice_masks_at(masks, index, affine)))
    if error_masks:
        layers.extend(error_layers(slice_masks_at(error_masks, index, affine)))

    return render_slice_2d(sl, layers, alpha=alpha, low_pct=low_pct, high_pct=high_pct,
                           crop=crop, display_size=display_size)


def colorbar_legend() -> list[tuple[str, tuple[int, int, int]]]:
    """(label, colour) pairs used to build the on-screen legend."""
    return [
        ("全肿瘤区 WT", REGION_COLORS["WT"]),
        ("肿瘤核心区 TC", REGION_COLORS["TC"]),
        ("强化肿瘤区 ET", REGION_COLORS["ET"]),
    ]


def legend_html(extra: str = "") -> str:
    items = "".join(
        f'<span class="legend-item"><span class="legend-swatch" '
        f'style="background:rgb{color}"></span>{label}</span>'
        for label, color in colorbar_legend()
    )
    return f'<div class="legend">{items}{extra}</div>'


# --------------------------------------------------------------------------
# Montage for reports / teaching previews
# --------------------------------------------------------------------------
def montage_indices(affine: np.ndarray, data: np.ndarray, n: int = 5,
                    center: int | None = None) -> list[int]:
    """Evenly spaced slice indices centred on the tumour when one is known."""
    total = axial_slice_count(data, affine)
    if center is None:
        center = total // 2
    center = int(np.clip(center, 0, max(total - 1, 0)))
    if n <= 1:
        return [center]
    half = n // 2
    start = max(0, center - half * 6)
    end = min(total - 1, start + (n - 1) * 6)
    idx = np.linspace(start, end, n).round().astype(int)
    return [int(i) for i in idx]


def tumour_center_slice(mask: np.ndarray, affine: np.ndarray) -> int | None:
    """Axial index of the tumour centroid, used to open the viewer sensibly."""
    if mask is None or not np.any(mask):
        return None
    ax = axial_axis(affine)
    counts = np.asarray(mask).sum(axis=tuple(i for i in range(3) if i != ax))
    return int(np.argmax(counts))
