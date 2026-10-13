#!/usr/bin/env python
"""Download the local Qwen3-0.6B weights (4-bit GGUF, ~462 MB).

The brief specifies ``qwen3:0.6b`` from the Ollama registry.  This host cannot
reach GitHub release assets, so the Ollama runtime could not be installed
without disrupting the shared environment; the project therefore serves the same
model through llama.cpp instead.  ``Qwen3-0.6B-Q4_K_M.gguf`` is the 4-bit
quantisation of the same weights (Ollama's ``qwen3:0.6b`` is ~523 MB).

    python scripts/fetch_local_model.py
    python scripts/fetch_local_model.py --list      # show available quantisations
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "models" / "qwen3-0.6b"

# Public mirrors, tried in order.  HuggingFace proper is unreachable from some
# networks; the mirror is tried first when HF_ENDPOINT says so.
MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]
REPO = "bartowski/Qwen_Qwen3-0.6B-GGUF"
#: File name as published in the repository.
REMOTE_FILE = "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
#: Name this project stores it under -- must match
#: ``local_llm_server.DEFAULT_MODEL`` or the server will not find it.
QUANT = "Qwen3-0.6B-Q4_K_M.gguf"


def list_quants() -> int:
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(f"{base}/api/models/{REPO}", timeout=40) as r:
                data = json.loads(r.read().decode())
            files = sorted(s["rfilename"] for s in data.get("siblings", [])
                           if s["rfilename"].endswith(".gguf"))
            print(f"来源: {base}/{REPO}")
            for f in files:
                mark = "  ← 默认" if f == REMOTE_FILE else ""
                print(f"  {f}{mark}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  镜像不可用 {base}: {type(exc).__name__}")
    return 1


def download(remote_file: str = REMOTE_FILE) -> int:
    """Fetch `remote_file` and store it under this project's canonical name."""
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / QUANT
    if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
        print(f"已存在，跳过下载: {target} "
              f"({target.stat().st_size / 1048576:.0f} MB)")
        return 0

    last: Exception | None = None
    for base in MIRRORS:
        url = f"{base}/{REPO}/resolve/main/{remote_file}"
        print(f"[download] {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "radiology-copilot/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(target, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            print(f"[download] 完成: {target} "
                  f"({target.stat().st_size / 1048576:.0f} MB)")
            return 0
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[download] 失败: {type(exc).__name__}: {exc}")
            target.unlink(missing_ok=True)
    print(f"所有镜像均失败: {last}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="列出可用量化版本")
    ap.add_argument("--file", default=REMOTE_FILE, help="仓库中的 GGUF 文件名")
    args = ap.parse_args()

    if args.list:
        return list_quants()
    if download(args.file) != 0:
        return 1
    print("\n下一步: python scripts/local_llm_server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
