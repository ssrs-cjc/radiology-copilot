"""Brain-tumour segmentation via the official MONAI BraTS bundle.

What this module does *not* do
------------------------------
It does not re-implement a network.  The architecture and the weights come from
the published bundle ``MONAI/brats_mri_segmentation`` and are loaded through
MONAI's own ``bundle.load`` API, so the thing doing the segmenting is the
released model, not a lookalike.

What this module does
---------------------
It reproduces the bundle's ``configs/inference.json`` pipeline so that a single
case can be segmented inside a web request.  Every step below was transcribed
from the bundle metadata that ships with the weights, not from memory:

    preprocessing   NormalizeIntensityd(nonzero=True, channel_wise=True)
    inferer         SlidingWindowInferer(roi_size=[240,240,160],
                                         sw_batch_size=1, overlap=0.5)
    postprocessing  sigmoid -> threshold 0.5 ->
                    ET(4) > TC(1) > WT(2) label remap

Input channels are ``T1c, T1, T2, FLAIR`` and output channels are
``TC, WT, ET`` -- note that whole tumour is channel **1**, which is an easy and
silent way to get the regions wrong, so it is asserted in a test.

If the weights are unavailable the module raises
:class:`SegmentationUnavailable`; callers fall back to the clearly-labelled
reference-annotation mode rather than inventing a prediction.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import metrics
from metrics import LabelConvention

PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = PROJECT_ROOT / "models" / "brats_mri_segmentation"

MODEL_NAME = "MONAI/brats_mri_segmentation"
MODEL_VERSION = "0.5.4"          # bundle version from configs/metadata.json
ARCHITECTURE = "SegResNet"

# Channel order the released weights were trained with.  Verified against the
# bundle's configs/metadata.json -> network_data_format.inputs.channel_def.
MODEL_INPUT_CHANNELS = ("t1ce", "t1", "t2", "flair")
# Output channel index -> region.  WT is index 1, not 0.
MODEL_OUTPUT_CHANNELS = {0: "TC", 1: "WT", 2: "ET"}
# Reverse lookup: region name -> channel index.
MODEL_OUTPUT_CHANNELS_REVERSE = {v: k for k, v in MODEL_OUTPUT_CHANNELS.items()}

# Sliding-window settings copied from configs/inference.json.
ROI_SIZE = (240, 240, 160)
SW_BATCH_SIZE = 1
SW_OVERLAP = 0.5

# Refuse to take a GPU with less than this much free memory, so we do not
# destabilise other people's jobs on a shared machine.
MIN_FREE_VRAM_BYTES = 6 * 1024 ** 3

BUNDLE_FILES = [
    "configs/inference.json",
    "configs/metadata.json",
    "models/model.pt",
]

MIRRORS = [
    "https://hf-mirror.com/{repo}/resolve/main/{path}",
    "https://huggingface.co/{repo}/resolve/main/{path}",
]


class SegmentationUnavailable(RuntimeError):
    """Raised when the released model cannot be used, with a user-facing reason."""


# --------------------------------------------------------------------------
# Bundle provisioning
# --------------------------------------------------------------------------
def bundle_present(bundle_dir: Path = BUNDLE_DIR) -> bool:
    return all((bundle_dir / f).exists() for f in BUNDLE_FILES)


def _download(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "radiology-copilot/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def ensure_bundle(bundle_dir: Path = BUNDLE_DIR, allow_download: bool = True) -> Path:
    """Make sure the released bundle is on disk. Returns its directory."""
    if bundle_present(bundle_dir):
        return bundle_dir
    if not allow_download:
        raise SegmentationUnavailable(
            f"未找到 MONAI 预训练权重 ({bundle_dir})。运行 "
            f"`python scripts/fetch_model.py` 下载，或保持参考标注模式。"
        )

    errors: list[str] = []
    for f in BUNDLE_FILES:
        target = bundle_dir / f
        if target.exists():
            continue
        for pattern in MIRRORS:
            url = pattern.format(repo=MODEL_NAME, path=f)
            try:
                _download(url, target)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        else:
            raise SegmentationUnavailable(
                "无法下载 MONAI 预训练权重。\n" + "\n".join(errors[-3:])
            )
    return bundle_dir


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
def _gpu_stats() -> list[tuple[int, int, int]]:
    """(index, free_MiB, utilisation_%) for every visible GPU.

    Read through NVML rather than torch so that we can see utilisation, not
    just memory.  This machine is shared: a GPU can have plenty of free VRAM
    while being pinned at 100% by somebody else's training job, and politely
    avoiding those is the difference between a 20-second inference and a
    10-minute one.
    """
    try:
        import pynvml
    except ImportError:
        return []
    stats: list[tuple[int, int, int]] = []
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            stats.append((i, int(mem.free // (1024 ** 2)), int(util.gpu)))
    except Exception:  # noqa: BLE001
        return stats
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return stats


def select_device(prefer: str | None = None) -> tuple[str, str]:
    """Pick an inference device without disturbing other work on this machine.

    Priority: explicit argument, then ``RADIOLOGY_DEVICE``, then the idle GPU
    with enough free VRAM, then CPU.  We deliberately fall back to CPU rather
    than crowding a machine that is already saturated.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu", "PyTorch 不可用，使用 CPU"

    if prefer:
        return prefer, f"用户指定 {prefer}"

    env = os.environ.get("RADIOLOGY_DEVICE")
    if env:
        return env, f"环境变量 RADIOLOGY_DEVICE={env}"

    if not torch.cuda.is_available():
        return "cpu", "未检测到可用 CUDA，使用 CPU"

    stats = _gpu_stats()
    if stats:
        usable = [s for s in stats if s[1] * 1024 ** 2 >= MIN_FREE_VRAM_BYTES]
        if usable:
            # Prefer idle devices; break ties by free memory.
            usable.sort(key=lambda s: (s[2], -s[1]))
            idx, free_mib, util = usable[0]
            note = f"选择 cuda:{idx}（剩余 {free_mib / 1024:.1f}GB, 利用率 {util}%）"
            if util > 50:
                note += "；注意该卡当前负载较高，推理可能变慢"
            return f"cuda:{idx}", note
        return (
            "cpu",
            f"所有 GPU 剩余显存均低于 {MIN_FREE_VRAM_BYTES / 1024 ** 3:.0f}GB，"
            f"为不影响他人任务改用 CPU",
        )

    # NVML unavailable: fall back to the torch memory query.
    best_idx, best_free = -1, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            continue
        if free > best_free:
            best_idx, best_free = i, free
    if best_idx < 0:
        return "cpu", "无法查询 GPU 显存，使用 CPU"
    if best_free < MIN_FREE_VRAM_BYTES:
        return "cpu", "剩余显存不足，改用 CPU"
    return f"cuda:{best_idx}", f"选择 cuda:{best_idx}（剩余 {best_free / 1024 ** 3:.1f}GB）"


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
# The post-processing rule applied to the model's three output channels.  Stored
# alongside every prediction so a mask can always be traced back to the rule
# that produced it.
POSTPROCESS_RULE = (
    "sigmoid → threshold 0.5 → label map with precedence ET(4) > TC(1) > WT(2)"
)


@dataclass
class NestingConflicts:
    """Voxels where the model violated ET ⊆ TC ⊆ WT.

    The released model occasionally emits a voxel as enhancing tumour (ET) while
    its tumour-core (TC) channel says background, or TC without WT.  Rather than
    silently trusting the channel order, we count these, record them, and encode
    by explicit precedence.  Keeping the count makes the behaviour auditable and
    lets the research module see how often it happens.
    """

    et_outside_tc: int = 0
    tc_outside_wt: int = 0
    postprocess_rule: str = POSTPROCESS_RULE

    @property
    def total(self) -> int:
        return self.et_outside_tc + self.tc_outside_wt

    def as_dict(self) -> dict:
        return {
            "et_outside_tc": self.et_outside_tc,
            "tc_outside_wt": self.tc_outside_wt,
            "total": self.total,
            "postprocess_rule": self.postprocess_rule,
        }

    def describe(self) -> str:
        return (
            f"模型输出的三个区域存在 {self.total} 个体素不满足 ET⊆TC⊆WT 嵌套关系"
            f"（ET∉TC: {self.et_outside_tc}，TC∉WT: {self.tc_outside_wt}），"
            f"原始预测已完整保留，仅按 ET>TC>WT 优先级编码"
        )


@dataclass
class SegmentationResult:
    """Everything a caller needs to display, measure and record a prediction."""

    label_map: np.ndarray                      # BraTS-encoded: 1=NCR/NET, 2=ED, 4=ET
    region_masks: dict[str, np.ndarray]        # WT / TC / ET binary masks
    channel_probabilities: np.ndarray | None   # raw sigmoid probabilities
    spacing: tuple[float, float, float]
    affine: np.ndarray
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    architecture: str = ARCHITECTURE
    device: str = "cpu"
    device_reason: str = ""
    duration_s: float = 0.0
    channel_order: tuple[str, ...] = MODEL_INPUT_CHANNELS
    warnings: list[str] = field(default_factory=list)
    nesting: NestingConflicts = field(default_factory=NestingConflicts)
    # Raw per-channel masks before the precedence encoding, kept so the original
    # prediction can always be reconstructed.
    raw_channel_masks: dict[str, np.ndarray] | None = None

    @property
    def volumes(self) -> dict[str, metrics.VolumeResult]:
        return metrics.all_region_volumes(self.region_masks, self.spacing)

    def provenance(self) -> str:
        return f"{self.model_name} v{self.model_version} ({self.architecture}) on {self.device}"

    def record(self) -> dict:
        """Auditable summary persisted with every analysis run."""
        return {
            "nesting_conflicts": self.nesting.as_dict(),
            "postprocess_rule": POSTPROCESS_RULE,
            "channel_order": list(self.channel_order),
            "threshold": 0.5,
        }


def _load_network(bundle_dir: Path, device: str):
    """Load the released architecture and weights through MONAI's own APIs.

    The architecture is not hard-coded here: it is instantiated from the
    ``network_def`` entry of the bundle's own ``configs/inference.json``, so the
    object we build is by construction the one the weights were trained for.
    ``copy_model_state`` reports how many tensors were matched, and we refuse to
    continue if any weight failed to load -- a partially initialised network
    would still produce confident-looking masks.
    """
    import json

    import torch
    from monai.bundle import ConfigParser

    # Never let MONAI reach for the network while serving a request.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    config_path = bundle_dir / "configs" / "inference.json"
    weights_path = bundle_dir / "models" / "model.pt"
    if not config_path.exists() or not weights_path.exists():
        raise SegmentationUnavailable(
            f"bundle 文件不完整: {config_path.exists()=} {weights_path.exists()=}"
        )

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        parser = ConfigParser(config=cfg)
        network = parser.get_parsed_content("network_def")
    except Exception as exc:  # noqa: BLE001
        raise SegmentationUnavailable(
            f"无法从 bundle 配置实例化网络: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise SegmentationUnavailable(f"无法读取模型权重: {type(exc).__name__}: {exc}") from exc

    # The released checkpoint is a flat state dict keyed exactly like the
    # network.  Strict loading means any structural mismatch raises instead of
    # leaving part of the network randomly initialised -- a failure mode that
    # would still emit confident-looking masks.
    try:
        network.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise SegmentationUnavailable(
            f"模型权重与网络结构不匹配，拒绝使用未完整加载的模型: {type(exc).__name__}: {exc}"
        ) from exc

    network = network.to(device)
    network.eval()
    return network


def segment_volume(
    image_4ch: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    affine: np.ndarray,
    convention: LabelConvention = metrics.BRATS_2020,
    device: str | None = None,
    bundle_dir: Path = BUNDLE_DIR,
    allow_download: bool = True,
    progress: Any = None,
) -> SegmentationResult:
    """Segment a 4-channel volume and return a BraTS-encoded label map.

    ``image_4ch`` must already be stacked in :data:`MODEL_INPUT_CHANNELS` order;
    use :func:`cases.order_channels` to build it so the ordering rule lives in
    exactly one place.
    """
    import torch
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import Activations, AsDiscrete, NormalizeIntensity

    arr = np.asarray(image_4ch, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"期望 4 通道输入 (X,Y,Z,4)，实际为 {arr.shape}")

    bundle_dir = ensure_bundle(bundle_dir, allow_download=allow_download)
    if device is None:
        device, reason = select_device()
    else:
        reason = f"调用方指定 {device}"

    if progress:
        progress(f"加载模型权重 ({ARCHITECTURE}) …")
    network = _load_network(bundle_dir, device)

    warnings: list[str] = []

    # --- preprocessing -----------------------------------------------------
    # The bundle specifies NormalizeIntensityd(nonzero=True, channel_wise=True).
    # We call MONAI's transform rather than re-deriving it, because the exact
    # semantics matter: with nonzero=True the z-score is applied *only* to the
    # non-zero voxels and background is left at zero.  Normalising the whole
    # volume instead drives background to about -4 sigma, which is a
    # distribution the network never saw during training and which is enough to
    # collapse its output to a fraction of the true lesion.
    if progress:
        progress("预处理: 逐通道非零区域标准化 (MONAI NormalizeIntensity) …")
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float()          # (4, X, Y, Z)
    for c in range(tensor.shape[0]):
        if float(tensor[c].abs().sum()) == 0:
            warnings.append(f"通道 {c} 全为零，标准化已跳过")
    tensor = NormalizeIntensity(nonzero=True, channel_wise=True)(tensor)
    tensor = tensor.unsqueeze(0).to(device)                              # (1, 4, X, Y, Z)

    # --- sliding-window inference -----------------------------------------
    if progress:
        progress(f"滑窗推理中 (roi={ROI_SIZE}, overlap={SW_OVERLAP}) …")
    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, overlap=SW_OVERLAP,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = inferer(tensor, network)
        probs = Activations(sigmoid=True)(logits)
    duration = time.perf_counter() - t0

    probs_chw = probs[0].detach().float().cpu()                 # (3, X, Y, Z)
    probs_np = probs_chw.numpy().transpose(1, 2, 3, 0)          # (X, Y, Z, 3) for storage

    # --- postprocessing: threshold, then ET > TC > WT remap ---------------
    if progress:
        progress("后处理: 阈值化与区域编码 …")
    disc = AsDiscrete(threshold=0.5)(probs_chw).numpy().astype(bool)   # (3, X, Y, Z)

    et_ch = disc[MODEL_OUTPUT_CHANNELS_REVERSE["ET"]]
    tc_ch = disc[MODEL_OUTPUT_CHANNELS_REVERSE["TC"]]
    wt_ch = disc[MODEL_OUTPUT_CHANNELS_REVERSE["WT"]]

    # Sanity: the three channels should be nested.  The released model can
    # violate this at the voxel level, so we count it, keep the raw channels and
    # report it instead of hiding it.
    nesting = NestingConflicts(
        et_outside_tc=int(np.logical_and(et_ch, ~tc_ch).sum()),
        tc_outside_wt=int(np.logical_and(tc_ch, ~wt_ch).sum()),
    )
    if nesting.total:
        warnings.append(nesting.describe())

    # The raw channel masks exactly as the network produced them.
    raw_channels = {"ET": et_ch.copy(), "TC": tc_ch.copy(), "WT": wt_ch.copy()}

    masks = {
        "ET": et_ch,
        "TC": tc_ch,
        "WT": wt_ch,
    }
    label_map = metrics.masks_to_label_map(
        masks,
        et_value=4 if convention is metrics.BRATS_2020 else convention.et_labels[0],
        tc_value=1,
        wt_value=2,
    )

    # Re-derive the masks from the encoded label map so that the displayed
    # masks and the stored label map can never disagree.
    canonical_masks = metrics.region_masks(label_map, convention)

    return SegmentationResult(
        label_map=label_map,
        region_masks=canonical_masks,
        channel_probabilities=probs_np,
        spacing=tuple(float(s) for s in spacing),  # type: ignore[arg-type]
        affine=np.asarray(affine),
        device=device,
        device_reason=reason,
        duration_s=duration,
        warnings=warnings,
        nesting=nesting,
        raw_channel_masks=raw_channels,
    )


# --------------------------------------------------------------------------
# Reference-annotation fallback
# --------------------------------------------------------------------------
@dataclass
class ReferenceAnnotation:
    """A dataset's expert mask, presented in the same shape as a prediction.

    The UI must always make clear that this is *not* model output.
    """

    label_map: np.ndarray
    region_masks: dict[str, np.ndarray]
    spacing: tuple[float, float, float]
    affine: np.ndarray
    source: str

    @property
    def volumes(self) -> dict[str, metrics.VolumeResult]:
        return metrics.all_region_volumes(self.region_masks, self.spacing)

    @property
    def provenance(self) -> str:
        return self.source


def load_reference_annotation(path: str | Path, convention: LabelConvention) -> ReferenceAnnotation:
    """Load a dataset reference mask and derive its region masks."""
    import imaging

    vol = imaging.load_volume(path)
    lm = np.asarray(vol.data)
    if lm.ndim == 4:
        lm = lm[..., 0]
    lm = lm.astype(np.uint8)
    return ReferenceAnnotation(
        label_map=lm,
        region_masks=metrics.region_masks(lm, convention),
        spacing=tuple(float(s) for s in vol.spacing),  # type: ignore[arg-type]
        affine=vol.affine,
        source="数据集参考标注 (Reference annotation, 非模型预测)",
    )


def model_status() -> dict[str, Any]:
    """Report model availability for the UI status strip."""
    present = bundle_present()
    device, reason = select_device()
    return {
        "available": present,
        "bundle_present": present,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE,
        "device": device,
        "device_reason": reason,
        "input_channels": list(MODEL_INPUT_CHANNELS),
        "output_regions": [MODEL_OUTPUT_CHANNELS[i] for i in sorted(MODEL_OUTPUT_CHANNELS)],
        "bundle_dir": str(BUNDLE_DIR),
    }
