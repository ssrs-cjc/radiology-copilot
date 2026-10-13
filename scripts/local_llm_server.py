#!/usr/bin/env python
"""Serve the local Qwen3-0.6B model with an OpenAI-compatible API.

Why llama.cpp rather than Ollama
--------------------------------
The brief asked for Ollama, but this server cannot reach GitHub release assets
(the Ollama binary is distributed only there), so Ollama could not be installed
without violating the "do not disturb other users' environment" constraint by
reaching for a system package.  ``llama-cpp-python`` is a pip-installable,
dependency-light equivalent that serves the *same* kind of endpoint:

    POST /v1/chat/completions      OpenAI-compatible, including `tools`
    POST /v1/completions
    GET  /v1/models

It runs as a plain user process, needs no daemon, no root, and no GPU, which
fits this shared machine better than an always-on system service would.

Usage
-----
    python scripts/local_llm_server.py            # foreground
    python scripts/local_llm_server.py --port 8080 --threads 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "qwen3-0.6b" / "Qwen3-0.6B-Q4_K_M.gguf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback only by default; nothing is exposed publicly")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4),
                    help="CPU threads; kept modest so other users' jobs are not starved")
    ap.add_argument("--ctx", type=int, default=8192,
                    help="context window; the tool schemas need headroom")
    ap.add_argument("--gpu-layers", type=int, default=0,
                    help="0 = pure CPU. Every GPU on this host is busy with "
                         "other users' jobs, so CPU is the polite default.")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"模型文件不存在: {args.model}", file=sys.stderr)
        print("请先运行: python scripts/fetch_local_model.py", file=sys.stderr)
        return 2

    # Imported lazily so --help works without the dependency installed.
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    print("=" * 74)
    print("  Radiology Copilot · 本地小模型服务 (llama.cpp, OpenAI 兼容)")
    print("=" * 74)
    print(f"  模型      : {args.model.name} "
          f"({args.model.stat().st_size / 1048576:.0f} MB)")
    print(f"  设备      : {'CPU' if args.gpu_layers == 0 else f'{args.gpu_layers} 层卸载到 GPU'}"
          f" · {args.threads} 线程")
    print(f"  上下文    : {args.ctx} tokens")
    print(f"  监听      : http://{args.host}:{args.port}  (仅本机)")
    print(f"  接口      : /v1/chat/completions  /v1/models")
    print("=" * 74, flush=True)

    model_settings = ModelSettings(
        model=str(args.model),
        model_alias="qwen3-0.6b",
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )
    app = create_app(server_settings=ServerSettings(host=args.host, port=args.port),
                     model_settings=[model_settings])
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
