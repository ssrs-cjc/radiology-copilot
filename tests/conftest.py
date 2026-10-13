"""Shared test fixtures.

Tests must pass on a machine with **no dataset and no model weights**, so
nothing here downloads anything or requires the BraTS subset to exist.  Cases
that need a real volume build a small NIfTI in a temp directory instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import metrics  # noqa: E402  (after sys.path fix)


# --------------------------------------------------------------------------
# Environment isolation
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_llm_environment(monkeypatch):
    """Keep the suite independent of whatever machine it runs on.

    A developer may have a real ``.env`` with a live API key (that is the normal
    working state).  Without this, ``load_llm_config`` would pick it up and any
    test asserting the *unconfigured* behaviour would pass or fail depending on
    whose laptop it ran on -- and a test run would risk making real API calls.
    """
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "RADIOLOGY_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False, raising=False)
    except ImportError:  # pragma: no cover - dotenv is an optional dependency
        pass
    yield


# --------------------------------------------------------------------------
# Synthetic label maps (not patient data - just small arrays for arithmetic)
# --------------------------------------------------------------------------
@pytest.fixture
def brats_label_map() -> np.ndarray:
    """A 12x12x6 label map containing all three BraTS labels.

    Layout (per axial slice): a 6x6 block of oedema (2) containing a 3x3 core
    block (1) which itself contains a 1x1 enhancing voxel (4).  This reproduces
    the nested structure the real data has.
    """
    lm = np.zeros((12, 12, 6), dtype=np.uint8)
    lm[3:9, 3:9, 1:5] = 2          # oedema          -> WT only
    lm[4:7, 4:7, 1:5] = 1          # NCR/NET core    -> WT + TC
    lm[5, 5, 1:5] = 4              # enhancing       -> WT + TC + ET
    return lm


@pytest.fixture
def spacing() -> tuple[float, float, float]:
    return (1.0, 1.0, 1.0)


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    d = tmp_path / "BraTS20_Training_999"
    d.mkdir(parents=True)
    return d


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    """Small helper so tests can create real NIfTI files on disk."""
    import nibabel as nib

    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))
    return path


@pytest.fixture
def synth_case(case_dir: Path):
    """A complete four-modality case on disk, with a reference mask."""
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [0.0, 11.0, 0.0]
    shape = (16, 16, 8)
    rng = np.random.default_rng(0)

    paths = {}
    for i, mod in enumerate(("t1", "t1ce", "t2", "flair")):
        data = np.zeros(shape, dtype=np.float32)
        data[2:14, 2:14, 1:7] = 100.0 + 10 * i
        data[6:10, 6:10, 2:6] = 300.0 + 40 * i     # "lesion" is brighter
        data += rng.normal(0, 2, shape)
        data[data < 0] = 0
        paths[mod] = str(write_nifti(case_dir / f"BraTS20_Training_999_{mod}.nii", data, affine))

    lm = np.zeros(shape, dtype=np.uint8)
    lm[6:10, 6:10, 2:6] = 2
    lm[7:9, 7:9, 2:6] = 1
    lm[7:9, 7:9, 3:5] = 4
    ref = str(write_nifti(case_dir / "BraTS20_Training_999_seg.nii", lm, affine))

    return {"case_id": "BraTS20_Training_999", "image_paths": paths,
            "reference_mask_path": ref, "label_map": lm, "affine": affine}
