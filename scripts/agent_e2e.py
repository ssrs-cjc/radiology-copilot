#!/usr/bin/env python
"""Real end-to-end LLM Agent run against a live API.

Runs the demo scenario from the Milestone 3 brief:

    请分析当前病例，查询脑肿瘤分割结果，整理区域体积，
    并生成一份待审核影像分析报告。

against a real OpenAI-compatible endpoint, executes the tools the model
chooses, and writes an **anonymised** execution record.

Privacy
-------
The record contains the model name, timings, the tool names/arguments/statuses
and the tool summaries.  It never contains the API key, the base URL, request
headers, filesystem paths, image data or anything patient-identifying.  The
tools themselves only ever return structured metadata about a public demo case.

Usage
-----
    export LLM_API_KEY=...        # never commit this
    export LLM_BASE_URL=https://api.deepseek.com
    export LLM_MODEL=deepseek-chat
    python scripts/agent_e2e.py
    python scripts/agent_e2e.py --dry-run     # check config only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

import agent as agent_mod  # noqa: E402
import cases as cases_mod  # noqa: E402
import imaging  # noqa: E402
import metrics  # noqa: E402
import report as report_mod  # noqa: E402
import segmentation  # noqa: E402
import storage  # noqa: E402

DEMO_PROMPT = (
    "请分析当前病例，查询脑肿瘤分割结果，整理区域体积，"
    "并生成一份待审核影像分析报告。"
)

RECORD_PATH = PROJECT_ROOT / "outputs" / "agent_e2e_record.json"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"


# --------------------------------------------------------------------------
# Session construction (no Gradio import, so this runs headless)
# --------------------------------------------------------------------------
def load_prediction(case_id: str) -> np.ndarray | None:
    import nibabel as nib

    p = PRED_DIR / f"{case_id}_pred.nii.gz"
    if not p.exists():
        return None
    return np.asarray(nib.load(str(p)).dataobj).astype(np.uint8)


def build_session(case_id: str, store) -> tuple[agent_mod.AgentSession, str]:
    """Wire a session around an existing prediction, or run segmentation."""
    registry = cases_mod.load_registry()
    case = registry.get(case_id)
    if case is None:
        raise SystemExit(f"病例 {case_id} 不存在。先运行 scripts/fetch_data.py。")

    base = imaging.load_volume(case.image_paths[case.modalities[0]])
    label_map = load_prediction(case_id)
    source = "已保存的模型预测"
    device, duration = "precomputed", 0.0

    if label_map is None:
        print("[e2e] 未找到已保存的预测，正在运行分割模型 …")
        volumes = {m: imaging.load_volume(case.image_paths[m]).data for m in case.modalities}
        result = segmentation.segment_volume(
            cases_mod.order_channels(case, volumes),
            spacing=base.spacing, affine=base.affine)
        label_map = result.label_map
        device, duration = result.device, result.duration_s
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        import nibabel as nib
        nib.save(nib.Nifti1Image(label_map.astype(np.uint8), base.affine),
                 str(PRED_DIR / f"{case_id}_pred.nii.gz"))
        source = "本次运行的模型预测"
    else:
        # Reuse the device from the run that produced the stored prediction, so
        # the agent reports how the mask was actually computed rather than a
        # placeholder.
        runs = store.list_analysis_runs(case_id, limit=1)
        if runs:
            device = runs[0].get("device") or device
            duration = runs[0].get("duration_s") or duration

    # Run the same input checks the UI runs, so `validate_mri` has a real
    # result instead of reporting that validation never happened.
    validation = imaging.validate_case(case.image_paths, case.reference_path())

    # Register the case as analysed, so `get_case_info` reports the true state.
    # Both the stored row and the in-memory Case are updated: tools read the
    # object, and leaving its field stale made the agent report the case as
    # never segmented while simultaneously quoting its segmentation.
    store.upsert_case(
        case_id,
        segmentation_status="completed",
        model_name=segmentation.MODEL_NAME,
        model_version=segmentation.MODEL_VERSION,
    )
    case.segmentation_status = "completed"
    case.model_name = segmentation.MODEL_NAME
    case.model_version = segmentation.MODEL_VERSION

    seg_result = segmentation.SegmentationResult(
        label_map=label_map,
        region_masks=metrics.region_masks(label_map, case.convention),
        channel_probabilities=None,
        spacing=base.spacing, affine=base.affine,
        device=device, duration_s=float(duration))

    reference = None
    evaluation = None
    if case.has_reference:
        ref_lm = imaging.load_volume(str(case.reference_path())).data.astype(np.uint8)
        reference = segmentation.ReferenceAnnotation(
            label_map=ref_lm, region_masks=metrics.region_masks(ref_lm, case.convention),
            spacing=base.spacing, affine=base.affine, source="数据集参考标注")
        evaluation = metrics.evaluate_segmentation(label_map, ref_lm, base.spacing,
                                                    case.convention)

    def _build_report() -> tuple[str, str]:
        ctx = report_mod.ReportContext(
            case_id=case.case_id,
            dataset_source=case.dataset_source,
            data_origin_label=case.origin_label,
            modalities=list(case.modalities),
            analysis_method="MONAI brats_mri_segmentation (SegResNet) 滑窗推理",
            segmentation_source="AI 模型预测",
            model_provenance=f"{segmentation.MODEL_NAME} v{segmentation.MODEL_VERSION}",
            volumes=metrics.all_region_volumes(seg_result.region_masks, base.spacing),
            spacing=base.spacing,
            shape=tuple(base.data.shape[:3]),
            review_status="unreviewed",
        )
        return report_mod.render_template(ctx), "template"

    def _save_report(content: str, generated_by: str) -> str:
        store.save_report(case.case_id, content, "pending_review", generated_by)
        return "pending_review"

    session = agent_mod.AgentSession(
        case=case, case_id=case.case_id, store=store,
        validation=validation,
        segmentation=seg_result, reference=reference, evaluation=evaluation,
        report_builder=_build_report, report_saver=_save_report,
    )
    return session, source


def assert_no_leakage(record: dict, cfg: agent_mod.LLMConfig) -> list[str]:
    """Verify the saved record contains no credential or path material."""
    blob = json.dumps(record, ensure_ascii=False)
    problems: list[str] = []
    if cfg.api_key and cfg.api_key in blob:
        problems.append("记录中包含 API Key")
    if cfg.base_url and cfg.base_url in blob:
        problems.append("记录中包含 Base URL")
    for marker in (str(PROJECT_ROOT), "/data1/", "sk-", "Bearer"):
        if marker in blob:
            problems.append(f"记录中包含敏感标记: {marker}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="BraTS20_Training_001")
    ap.add_argument("--prompt", default=DEMO_PROMPT)
    ap.add_argument("--dry-run", action="store_true", help="check config and exit")
    ap.add_argument("--out", type=Path, default=RECORD_PATH)
    ap.add_argument("--local", action="store_true",
                    help="用本地 Qwen3-0.6B 端点（llama.cpp, OpenAI 兼容）替代 DeepSeek")
    ap.add_argument("--local-port", type=int, default=8080)
    ap.add_argument("--local-model", default="qwen3-0.6b")
    args = ap.parse_args()

    if args.local:
        cfg = agent_mod.LLMConfig(
            api_key="local-no-key", base_url=f"http://127.0.0.1:{args.local_port}/v1",
            model=args.local_model, enabled=True)
    else:
        cfg = agent_mod.load_llm_config()
    print("=" * 74)
    print("  Radiology Copilot · LLM Agent 端到端验证")
    print("=" * 74)
    print(f"  配置状态 : {cfg.describe()}")

    if not cfg.enabled:
        print("\n  ⚠ 未配置语言模型 API，无法执行真实端到端验证。\n")
        print("  请设置以下环境变量后重试（不要提交到仓库）：\n")
        print("      export LLM_API_KEY=<your-key>")
        print("      export LLM_BASE_URL=https://api.deepseek.com")
        print("      export LLM_MODEL=deepseek-chat\n")
        print("  或以固定工作流模式使用界面：python app.py\n")
        return 2

    if args.dry_run:
        ok, reason = agent_mod.probe_tool_calling(cfg)
        print(f"  函数调用探测 : {reason}")
        print(f"  结论         : {'可使用真实工具调用模式' if ok else '将使用固定工作流模式'}")
        return 0 if ok else 1

    store = storage.Store()
    session, pred_source = build_session(args.case, store)
    print(f"  病例         : {session.case.case_id}（{session.case.origin_label}）")
    print(f"  分割结果来源 : {pred_source}")
    print(f"  提问         : {args.prompt}")

    ok, probe_reason = agent_mod.probe_tool_calling(cfg)
    print(f"  函数调用探测 : {probe_reason}")
    if not ok:
        print("\n  ⚠ 该端点未通过函数调用探测，无法完成真实工具调用验证。")
        return 1

    print("-" * 74)
    turn = agent_mod.run_agent_turn(args.prompt, session, cfg, tools_supported=True)

    print(f"  执行模式     : {turn.mode}")
    print(f"  模型         : {turn.model}")
    print(f"  工具调用     : {len(turn.tool_calls)} 次")
    for i, step in enumerate(turn.tool_calls, 1):
        ms = step.get("duration_ms")
        timing = f"{ms:6.0f} ms" if isinstance(ms, (int, float)) else "     —  "
        print(f"    {i}. [{step['status']:7s}] {step['tool']:28s} {timing}  {step['summary']}")
        if step.get("arguments"):
            print(f"         参数: {json.dumps(step['arguments'], ensure_ascii=False)}")
    print(f"  模型调用     : {turn.model_calls} 次 API 往返")
    print(f"  模型耗时     : {turn.thinking_ms / 1000:.2f} s")
    print(f"  工具耗时     : {turn.tool_time_ms / 1000:.2f} s")
    print(f"  总计         : {turn.total_ms / 1000:.2f} s")
    print(f"  一致性校验   : {'通过' if turn.consistency.ok else turn.consistency.problems}")
    if turn.error:
        print(f"  错误         : {turn.error}")
    print("-" * 74)
    print("  最终回答：")
    for line in (turn.answer or "（无回答）").splitlines():
        print(f"    {line}")

    draft = turn.results.get("generate_report_draft")
    if draft and draft.status == "ok":
        print("-" * 74)
        print(f"  报告草稿     : 已生成，状态 {draft.data.get('status')}，"
              f"{draft.data.get('characters')} 字符，需人工审核={draft.data.get('requires_human_review')}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": session.case.case_id,
        "dataset": session.case.dataset_source,
        "data_origin": session.case.origin_label,
        "segmentation_source": pred_source,
        "model_config": {"model": cfg.model, "provider": "openai-compatible"},
        "probe": probe_reason,
        "turn": turn.as_record(),
        "report_revision": store.latest_report(session.case.case_id),
    }
    # The revision's content is fine to keep; strip nothing else.
    if record["report_revision"]:
        record["report_revision"] = {
            k: record["report_revision"].get(k)
            for k in ("revision", "status", "generated_by", "created_at")
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
