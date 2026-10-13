#!/usr/bin/env python
"""Evaluate the released MONAI model against expert annotation on every local case.

This produces the numbers quoted in ``docs/EVALUATION.md``.  It runs the real
model, on the real public cases, and reports the real Dice -- nothing here is
estimated or illustrative.

    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --json outputs/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cases as cases_mod  # noqa: E402
import imaging  # noqa: E402
import metrics  # noqa: E402
import segmentation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="also write raw results here")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    registry = cases_mod.load_registry(Path(args.data_dir) if args.data_dir else None)
    targets = [c for c in registry.cases.values() if c.has_reference]
    if not targets:
        print("No cases with reference annotation found. "
              "Run `python scripts/fetch_data.py --cases 6` first.")
        return 1

    device, reason = segmentation.select_device(args.device)
    print(f"device: {device}  ({reason})")
    print(f"model : {segmentation.MODEL_NAME} v{segmentation.MODEL_VERSION} "
          f"({segmentation.ARCHITECTURE})")
    print(f"cases : {len(targets)}")
    print("-" * 92)
    print(f"{'case':<26} {'WT':>8} {'TC':>8} {'ET':>8} {'mean':>8} {'time':>7}  {'WT mL':>8}")
    print("-" * 92)

    per_case: list[dict] = []
    for case in targets:
        volumes = {m: imaging.load_volume(case.image_paths[m]).data for m in case.modalities}
        base = imaging.load_volume(case.image_paths[case.modalities[0]])
        image_4ch = cases_mod.order_channels(case, volumes)

        t0 = time.time()
        result = segmentation.segment_volume(
            image_4ch, spacing=base.spacing, affine=base.affine,
            device=device, convention=case.convention)
        elapsed = time.time() - t0

        ref = imaging.load_volume(str(case.reference_path())).data.astype("uint8")
        evaluation = metrics.evaluate_segmentation(
            result.label_map, ref, base.spacing, case.convention)

        dice = {r: evaluation.dice[r].dice for r in ("WT", "TC", "ET")}
        mean_dice = evaluation.mean_dice()
        wt_ml = result.volumes["WT"].volume_ml

        print(f"{case.case_id:<26} {dice['WT']:>8.4f} {dice['TC']:>8.4f} {dice['ET']:>8.4f} "
              f"{mean_dice:>8.4f} {elapsed:>6.1f}s  {wt_ml:>8.2f}")

        per_case.append({
            "case_id": case.case_id,
            "dataset_source": case.dataset_source,
            "label_convention": case.label_convention,
            "device": result.device,
            "duration_s": round(elapsed, 2),
            "dice": dice,
            "mean_dice": mean_dice,
            "missed_voxels": evaluation.missed,
            "extra_voxels": evaluation.extra,
            "predicted_volumes_ml": {r: v.volume_ml for r, v in result.volumes.items()},
            "warnings": result.warnings,
        })

    print("-" * 92)
    for region in ("WT", "TC", "ET"):
        vals = [c["dice"][region] for c in per_case if c["dice"][region] is not None]
        print(f"mean {region} Dice: {statistics.mean(vals):.4f}  "
              f"(sd {statistics.pstdev(vals):.4f}, n={len(vals)})")
    means = [c["mean_dice"] for c in per_case if c["mean_dice"] is not None]
    print(f"mean of per-case mean Dice: {statistics.mean(means):.4f}")
    print(f"total inference time: {sum(c['duration_s'] for c in per_case):.1f}s "
          f"for {len(per_case)} cases")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "model": segmentation.MODEL_NAME,
            "model_version": segmentation.MODEL_VERSION,
            "architecture": segmentation.ARCHITECTURE,
            "input_channels": list(segmentation.MODEL_INPUT_CHANNELS),
            "output_regions": [segmentation.MODEL_OUTPUT_CHANNELS[i]
                               for i in sorted(segmentation.MODEL_OUTPUT_CHANNELS)],
            "device": device,
            "cases": per_case,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
