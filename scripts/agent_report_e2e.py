#!/usr/bin/env python
"""Real end-to-end test of the LLM-written report narrative.

Milestone 4, task 2.  Runs the *report narra* path (not the tool-calling path)
against a live OpenAI-compatible endpoint and checks the result.

What is verified
----------------
1. The model receives only **structured facts** -- case id, modalities, volumes,
   model provenance.  No image data is ever sent.
2. The narrative is merged into the deterministic template, so every number in
   the final report comes from the computed values, not from the model.
3. The narrative is re-checked against those values afterwards: the case id must
   match, every quoted volume must match, and no pathology / grading / treatment
   language may appear.
4. The saved report stays ``pending_review``; the model cannot approve it.
5. An anonymised record is written (model, timings, token usage, checks).  The
   API key, base URL and filesystem paths are never written.

Usage
-----
    export LLM_API_KEY=...
    export LLM_BASE_URL=https://api.deepseek.com
    export LLM_MODEL=deepseek-chat
    python scripts/agent_report_e2e.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agent as agent_mod  # noqa: E402
import cases as cases_mod  # noqa: E402
import imaging  # noqa: E402
import metrics  # noqa: E402
import report as report_mod  # noqa: E402
import segmentation  # noqa: E402
import storage  # noqa: E402
from agent_e2e import assert_no_leakage, load_prediction  # noqa: E402

RECORD_PATH = PROJECT_ROOT / "outputs" / "agent_report_e2e_record.json"


def build_context(case) -> tuple[report_mod.ReportContext, dict]:
    """Assemble the report context from real computed values."""
    base = imaging.load_volume(case.image_paths[case.modalities[0]])
    label_map = load_prediction(case.case_id)
    if label_map is None:
        raise SystemExit(
            f"未找到 {case.case_id} 的模型预测。先运行 scripts/evaluate_all.py "
            f"或 python scripts/agent_e2e.py。")

    masks = metrics.region_masks(label_map, case.convention)
    volumes = metrics.all_region_volumes(masks, base.spacing)

    ctx = report_mod.ReportContext(
        case_id=case.case_id,
        dataset_source=case.dataset_source,
        data_origin_label=case.origin_label,
        modalities=list(case.modalities),
        analysis_method="MONAI brats_mri_segmentation (SegResNet) 滑窗推理，"
                        "逐通道非零区域标准化，sigmoid 阈值 0.5，区域按 ET>TC>WT 编码",
        segmentation_source="AI 模型预测",
        model_provenance=f"{segmentation.MODEL_NAME} v{segmentation.MODEL_VERSION}",
        volumes=volumes,
        spacing=base.spacing,
        shape=tuple(base.data.shape[:3]),
        review_status="unreviewed",
    )
    return ctx, {r: v.volume_ml for r, v in volumes.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="BraTS20_Training_001")
    ap.add_argument("--out", type=Path, default=RECORD_PATH)
    ap.add_argument("--dry-run", action="store_true", help="check config and exit")
    args = ap.parse_args()

    cfg = agent_mod.load_llm_config()
    print("=" * 74)
    print("  Radiology Copilot · LLM 报告叙述端到端验证")
    print("=" * 74)
    print(f"  配置状态 : {cfg.describe()}")

    if not cfg.enabled:
        print("\n  ⚠ 未配置语言模型 API，无法执行真实验证。")
        print("  请设置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 后重试。\n")
        return 2
    if args.dry_run:
        ok, reason = agent_mod.probe_tool_calling(cfg)
        print(f"  连通性   : {reason}")
        return 0 if ok else 1

    registry = cases_mod.load_registry()
    case = registry.get(args.case)
    if case is None:
        raise SystemExit(f"病例 {args.case} 不存在")

    ctx, computed_volumes = build_context(case)
    print(f"  病例     : {case.case_id}（{case.origin_label}）")
    print(f"  计算体积 : WT {computed_volumes['WT']:.2f} / "
          f"TC {computed_volumes['TC']:.2f} / ET {computed_volumes['ET']:.2f} mL")
    print(f"  发送内容 : 仅结构化字段（无影像、无路径、无患者信息）")

    # ---- what exactly goes to the model --------------------------------
    facts = report_mod.facts_for_llm(ctx)
    sent = json.dumps(facts, ensure_ascii=False)
    for marker in ("/data1/", ".nii", str(PROJECT_ROOT)):
        assert marker not in sent, f"事实集中不应包含 {marker}"
    print(f"  事实集   : {len(sent)} 字符，字段 {sorted(facts)}")

    print("-" * 74)
    outcome = agent_mod.generate_llm_report_sections(ctx, cfg)
    print(f"  调用结果 : {'成功' if outcome.ok else '失败'}")
    print(f"  模型     : {outcome.model}")
    print(f"  耗时     : {outcome.duration_ms / 1000:.2f} s")
    if outcome.usage_reported:
        print(f"  Token    : 输入 {outcome.prompt_tokens} / 输出 "
              f"{outcome.completion_tokens} / 合计 {outcome.total_tokens}")
    else:
        print("  Token    : 该 API 响应未返回 usage 字段")
    if not outcome.ok:
        print(f"  说明     : {outcome.status}")
        return 1

    # ---- consistency + safety check ------------------------------------
    check = agent_mod.verify_report_narrative(outcome.sections, ctx)
    print(f"  一致性   : {'通过' if check.ok else '发现问题'}")
    print(f"             核对体积 {check.checked_volumes}，"
          f"病例编号 {check.checked_case_ids}")
    for problem in check.problems:
        print(f"             ✗ {problem}")

    # ---- assemble the report the way the UI does ------------------------
    template = report_mod.render_template(ctx)
    final = report_mod.merge_llm_into_template(template, outcome.sections, outcome.model)

    numbers_present = all(
        f"{computed_volumes[r]:.2f} mL" in final for r in ("WT", "TC", "ET"))
    print(f"  草稿数值 : {'三个区域体积均与计算结果一致' if numbers_present else '⚠ 数值缺失'}")

    store = storage.Store()
    store.upsert_case(case.case_id, dataset_source=case.dataset_source)
    store.save_report(case.case_id, final, "pending_review", f"llm:{outcome.model}")
    revision = store.latest_report(case.case_id)
    print(f"  报告状态 : {revision['status']}（需人工审核，模型无法自行批准）"
          f" · 修订 r{revision['revision']}")

    print("-" * 74)
    print("  LLM 叙述部分：")
    for line in outcome.sections.splitlines():
        print(f"    {line}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case.case_id,
        "dataset": case.dataset_source,
        "task": "llm_report_narrative",
        "sent_facts_keys": sorted(facts),
        "sent_facts_chars": len(sent),
        "outcome": outcome.as_record(),
        "consistency": check.as_dict(),
        "final_report_numbers_match": numbers_present,
        "report_revision": {
            "revision": revision["revision"],
            "status": revision["status"],
            "generated_by": revision["generated_by"],
        },
        "narrative": outcome.sections,
    }

    leaks = assert_no_leakage(record, cfg)
    if leaks:
        print("\n  ✗ 脱敏检查未通过，未写出记录：")
        for problem in leaks:
            print(f"      - {problem}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 74)
    print(f"  脱敏执行记录 : {args.out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
