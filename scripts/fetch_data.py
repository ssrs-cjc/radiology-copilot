#!/usr/bin/env python
"""Fetch a small subset of real, public BraTS brain-tumour MRI cases.

Why this script exists
----------------------
The full public archives are large (MSD Task01_BrainTumour ~7.1 GB,
BraTS-2020 mirror ~4.5 GB) and this server reaches them at roughly 400 KB/s.
Downloading a whole archive would take hours.

The BraTS-2020 HuggingFace mirror is shipped as a single **zip** archive.
ZIP files carry a *central directory* at the very end that lists every member
and its byte offset. We can therefore:

  1. HTTP range-read only the last few hundred KB to obtain that directory,
  2. pick a handful of patients,
  3. range-read only the byte ranges belonging to those patients' files.

Net effect: we download ~300 MB instead of 4.5 GB, and we obtain genuine
BraTS data with genuine reference segmentation labels.

The archive is treated strictly as untrusted data: nothing here is executed.
Files are written under ``data/raw/`` and only ever parsed by NiBabel.

Usage
-----
    python scripts/fetch_data.py --list                  # show archive contents
    python scripts/fetch_data.py --cases 8               # fetch 8 patients
    python scripts/fetch_data.py --cases 4 --dry-run     # report sizes only
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "data" / "raw"

# Public mirrors, tried in order.  The HuggingFace endpoint can be redirected
# to a reachable mirror through HF_ENDPOINT (see .env.example).
CANDIDATE_ARCHIVES = [
    ("https://huggingface.co/datasets/Babai12345/BRATS-2020/resolve/main/BRATS-2020.zip",
     "BRATS-2020.zip"),
    ("https://hf-mirror.com/datasets/Babai12345/BRATS-2020/resolve/main/BRATS-2020.zip",
     "BRATS-2020.zip"),
]


class HttpRangeFile(io.RawIOBase):
    """A minimal seekable read-only file object backed by HTTP range requests.

    ``zipfile.ZipFile`` only needs ``read``/``seek``/``tell``; providing those
    lets the standard library parse a remote archive as if it were local.
    """

    def __init__(self, url: str, block_size: int = 1 << 20):
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self._size: int | None = None

    # -- helpers ---------------------------------------------------------
    def _request(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "radiology-copilot/0.1"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    @property
    def size(self) -> int:
        if self._size is None:
            req = urllib.request.Request(
                self.url, method="HEAD",
                headers={"User-Agent": "radiology-copilot/0.1"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                length = resp.headers.get("Content-Length")
                # HuggingFace redirects report the real size in x-linked-size.
                linked = resp.headers.get("x-linked-size")
                self._size = int(linked) if linked else int(length)
        return self._size

    # -- io API ----------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self._request(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:  # pragma: no cover - trivial
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_archive(dest: Path, allow_download: bool = True):
    """Return (zipfile.ZipFile, source_description)."""
    local = dest / "BRATS-2020.zip"
    if local.exists() and zipfile.is_zipfile(local):
        return zipfile.ZipFile(local), f"local file {local}"

    if not allow_download:
        raise SystemExit("No local archive available and downloads are disabled.")

    last_err: Exception | None = None
    for url, _name in CANDIDATE_ARCHIVES:
        try:
            print(f"[fetch] probing {url}", flush=True)
            rf = HttpRangeFile(url)
            size = rf.size
            print(f"[fetch] remote archive size = {size / 1e9:.2f} GB", flush=True)
            zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
            _ = zf.namelist()  # forces the central directory to be read
            return zf, url
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"[fetch] mirror failed: {type(exc).__name__}: {exc}", flush=True)
            last_err = exc
    raise SystemExit(f"All mirrors failed. Last error: {last_err}")


# Modality suffixes we want per patient, mapped to the canonical names used
# inside this project.  BraTS-2020 uses: flair / t1 / t1ce / t2 + seg.
WANTED_SUFFIXES = ("flair", "t1", "t1ce", "t2", "seg")


def classify(name: str):
    """Return (patient_id, suffix) for an archive member, or (None, None)."""
    base = name.rsplit("/", 1)[-1]
    if not base.endswith(".nii") and not base.endswith(".nii.gz"):
        return None, None
    stem = base
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # e.g. "BraTS20_Training_001_flair"
    for suffix in WANTED_SUFFIXES:
        if stem.endswith("_" + suffix):
            patient = stem[: -(len(suffix) + 1)]
            # patient looks like ".../BraTS20_Training_001"
            patient = patient.rsplit("/", 1)[-1]
            return patient, suffix
    return None, None


def group_by_patient(zf: zipfile.ZipFile) -> dict[str, dict[str, zipfile.ZipInfo]]:
    table: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        patient, suffix = classify(info.filename)
        if patient is None:
            continue
        table.setdefault(patient, {})[suffix] = info
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--cases", type=int, default=8, help="how many patients to fetch")
    ap.add_argument("--list", action="store_true", help="list archive contents and exit")
    ap.add_argument("--dry-run", action="store_true", help="report sizes without downloading")
    ap.add_argument("--no-download", action="store_true", help="refuse to touch the network")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    zf, source = open_archive(args.dest, allow_download=not args.no_download)
    print(f"[fetch] archive source: {source}", flush=True)

    patients = group_by_patient(zf)
    complete = {p: f for p, f in patients.items() if set(WANTED_SUFFIXES) <= set(f)}
    print(f"[fetch] patients in archive: {len(patients)}; complete cases: {len(complete)}")

    if args.list:
        for pid in sorted(complete)[:20]:
            files = complete[pid]
            total = sum(files[s].file_size for s in WANTED_SUFFIXES)
            print(f"  {pid:28s} files={len(files)} raw={total / 1e6:7.1f} MB")
        print(f"  ... ({len(complete)} total)")
        return 0

    selected = sorted(complete)[: args.cases]
    if not selected:
        raise SystemExit("No complete patients found in the archive.")

    out_root = args.dest / "brats_subset"
    grand_total = sum(complete[p][s].file_size for p in selected for s in WANTED_SUFFIXES)
    print(f"[fetch] selected {len(selected)} patients, {grand_total / 1e6:.1f} MB compressed")

    if args.dry_run:
        for pid in selected:
            print(f"  would fetch {pid}")
        return 0

    manifest = []
    for i, pid in enumerate(selected, 1):
        pdir = out_root / pid
        pdir.mkdir(parents=True, exist_ok=True)
        entry = {"patient_id": pid, "files": {}}
        for suffix in WANTED_SUFFIXES:
            info = complete[pid][suffix]
            target = pdir / os.path.basename(info.filename)
            if target.exists() and target.stat().st_size == info.file_size:
                print(f"  [{i}/{len(selected)}] {pid} {suffix:6s} cached", flush=True)
            else:
                t0 = time.time()
                with zf.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 18)
                        if not chunk:
                            break
                        dst.write(chunk)
                dt = time.time() - t0
                rate = info.file_size / max(dt, 1e-6) / 1024
                print(f"  [{i}/{len(selected)}] {pid} {suffix:6s} "
                      f"{info.file_size / 1e6:5.1f} MB in {dt:5.1f}s ({rate:5.0f} KB/s)", flush=True)
            entry["files"][suffix] = str(target.relative_to(PROJECT_ROOT))
        manifest.append(entry)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fetch] wrote {out_root / 'manifest.json'}")
    print(f"[fetch] done: {len(manifest)} cases under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
