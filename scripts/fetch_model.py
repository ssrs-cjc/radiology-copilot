#!/usr/bin/env python
"""Download the published MONAI BraTS segmentation bundle.

Fetches only the three files needed for inference (the network configuration,
its metadata and the trained weights) from the official
``MONAI/brats_mri_segmentation`` bundle repository.

The HuggingFace endpoint is not reachable from every network; set
``HF_ENDPOINT`` (see ``.env.example``) or rely on the built-in mirror fallback.

    python scripts/fetch_model.py
    python scripts/fetch_model.py --verify      # load the weights, no download
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import segmentation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="only check that the bundle loads; do not download")
    ap.add_argument("--device", default=None, help="device for the verification load")
    args = ap.parse_args()

    bundle_dir = segmentation.BUNDLE_DIR
    print(f"[model] bundle target: {bundle_dir}")

    if args.verify:
        if not segmentation.bundle_present(bundle_dir):
            print("[model] bundle NOT present")
            return 1
        print("[model] bundle files present:")
        for f in segmentation.BUNDLE_FILES:
            p = bundle_dir / f
            print(f"         {f:28s} {p.stat().st_size / 1e6:8.2f} MB")
        device, reason = segmentation.select_device(args.device)
        print(f"[model] device: {device}  ({reason})")
        t0 = time.time()
        net = segmentation._load_network(bundle_dir, device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[model] loaded {type(net).__name__} with {n_params:,} parameters "
              f"in {time.time() - t0:.1f}s")
        return 0

    t0 = time.time()
    try:
        segmentation.ensure_bundle(bundle_dir)
    except segmentation.SegmentationUnavailable as exc:
        print(f"[model] FAILED: {exc}")
        return 1

    print(f"[model] downloaded in {time.time() - t0:.1f}s")
    for f in segmentation.BUNDLE_FILES:
        p = bundle_dir / f
        print(f"         {f:28s} {p.stat().st_size / 1e6:8.2f} MB")
    print("\n[model] Next: python scripts/fetch_model.py --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
