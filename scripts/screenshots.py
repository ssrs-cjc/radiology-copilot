#!/usr/bin/env python
"""Capture screenshots from the running application.

These are produced by driving the real Gradio UI in a real browser, so the
images in ``assets/screenshots/`` show the program as it actually renders at
the specified 1440x900 viewport.

    python scripts/screenshots.py                 # app must already be running
    python scripts/screenshots.py --url http://127.0.0.1:7860 --run-segmentation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}


def settle(page: Page, ms: int = 900) -> None:
    page.wait_for_timeout(ms)


def wait_for_images(page: Page, timeout_ms: int = 45_000) -> None:
    """Wait until Gradio has finished rendering its image panels.

    Gradio shows a small "processing" overlay while a callback runs; capturing
    before it clears yields a blank image panel.
    """
    deadline = timeout_ms
    step = 500
    while deadline > 0:
        try:
            busy = page.locator("text=processing").count()
        except Exception:  # noqa: BLE001
            busy = 0
        if busy == 0:
            break
        page.wait_for_timeout(step)
        deadline -= step
    page.wait_for_timeout(1200)


def click_nav(page: Page, label: str) -> bool:
    """Click a left-nav entry. Returns False if the button is not present."""
    try:
        page.get_by_role("button", name=label, exact=False).first.click(timeout=8000)
        settle(page, 1200)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    ! nav '{label}' failed: {type(exc).__name__}")
        return False


def shot(page: Page, name: str) -> Path:
    wait_for_images(page)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"    ✓ {path.name}  ({path.stat().st_size // 1024} KB)")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--run-segmentation", action="store_true",
                    help="click through the segmentation flow first (~40s)")
    ap.add_argument("--include-teaching-submit", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        # channel="chromium" uses the full browser build; the lightweight
        # chrome-headless-shell is not always fetched by `playwright install`.
        browser = pw.chromium.launch(
            channel="chromium", args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        # Gradio keeps a websocket open forever, so "networkidle" never fires.
        # Wait for the app's own markup instead.
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector(".rc-topbar", timeout=120_000)
        page.wait_for_selector("button:has-text('智能阅片')", timeout=120_000)
        settle(page, 4000)

        print("  capturing pages")
        shot(page, "01_dashboard")

        if click_nav(page, "智能阅片"):
            settle(page, 2500)
            shot(page, "02_viewer_reference_annotation")

            if args.run_segmentation:
                print("    running AI segmentation through the UI …")
                try:
                    page.get_by_role("button", name="运行 AI 分割", exact=False).first.click(timeout=10_000)
                    # Sliding-window inference takes ~5-30 s depending on the GPU.
                    for _ in range(40):
                        page.wait_for_timeout(3000)
                        body = page.inner_text("body")
                        if "分割完成" in body or "分割失败" in body or "模型不可用" in body:
                            break
                    settle(page, 1500)
                    shot(page, "03_viewer_ai_segmentation")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! segmentation click failed: {type(exc).__name__}: {exc}")

        if click_nav(page, "AI 报告"):
            settle(page, 1000)
            try:
                page.get_by_role("button", name="生成报告草稿（模板）", exact=False).first.click(timeout=8000)
                settle(page, 2500)
                shot(page, "05_report_preview")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! report generation failed: {type(exc).__name__}")
                shot(page, "05_report_preview")

            # The edit tab is where the reviewer changes the wording.
            try:
                page.get_by_role("tab", name="编辑", exact=False).first.click(timeout=8000)
                settle(page, 1200)
                shot(page, "05b_report_edit")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! report edit tab failed: {type(exc).__name__}")

            # Drive the agent so the trace panel shows real tool execution.
            try:
                box = page.get_by_placeholder("例如：分析当前病例").first
                box.fill("请分析当前病例，查询脑肿瘤分割结果，整理区域体积，"
                         "并生成一份待审核影像分析报告。")
                page.get_by_role("button", name="发送", exact=False).first.click(timeout=8000)
                settle(page, 3500)
                shot(page, "06_report_agent_tools")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! agent interaction failed: {type(exc).__name__}")

        if click_nav(page, "教学模式"):
            settle(page, 1200)
            try:
                page.get_by_role("button", name="加载练习", exact=False).first.click(timeout=8000)
                settle(page, 3000)
                shot(page, "07_teaching_answer_hidden")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! teaching load failed: {type(exc).__name__}")
                shot(page, "07_teaching_answer_hidden")

            # Prove the reveal button cannot be used to skip the exercise.
            try:
                page.get_by_role("button", name="显示参考标注", exact=False).first.click(timeout=8000)
                settle(page, 2500)
                shot(page, "07b_teaching_reveal_blocked")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! reveal-block check failed: {type(exc).__name__}")

            if args.include_teaching_submit:
                try:
                    page.get_by_placeholder("描述观察到的异常区域").first.fill(
                        "FLAIR 序列见右侧额叶大片高信号，累及皮层下白质；"
                        "T1ce 可见结节状强化，考虑肿瘤核心区。")
                    page.get_by_role("button", name="提交答案", exact=False).first.click(timeout=8000)
                    settle(page, 3000)
                    # Switch to the feedback tab so the released answer and the
                    # grading breakdown are what the screenshot shows.
                    page.get_by_role("tab", name="反馈结果", exact=False).first.click(timeout=8000)
                    settle(page, 2000)
                    shot(page, "08_teaching_feedback")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! teaching submit failed: {type(exc).__name__}")

        if click_nav(page, "科研反馈"):
            settle(page, 1800)
            shot(page, "09_research_dashboard")

            # Human-correction loop.  The demo button replays the dataset
            # reference annotation as a stand-in correction, which is entirely
            # self-contained and is labelled as a demo everywhere it appears.
            try:
                page.get_by_text("人工修正掩膜回流", exact=False).first.click(timeout=8000)
                settle(page, 1200)
                page.get_by_role("button", name="演示：用参考标注模拟",
                                 exact=False).first.click(timeout=15000)
                settle(page, 4000)
                page.get_by_role("button", name="演示：用参考标注模拟",
                                 exact=False).first.scroll_into_view_if_needed(timeout=8000)
                page.evaluate("window.scrollBy(0, -120)")
                settle(page, 1500)
                shot(page, "09c_correction_diff")

                page.get_by_role("button", name="保存为科研反馈", exact=False).first.click(timeout=15000)
                settle(page, 4000)
                page.evaluate("window.scrollBy(0, 260)")
                settle(page, 1500)
                shot(page, "09d_correction_history")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! correction flow failed: {type(exc).__name__}: {exc}")
            # Expand the error-comparison panel and render it.
            try:
                page.get_by_text("误差可视化", exact=False).first.click(timeout=8000)
                settle(page, 1200)
                page.get_by_role("button", name="生成误差对比图", exact=False).first.click(timeout=15000)
                settle(page, 3500)
                # The panel sits below the statistics, so scroll it into frame.
                # (scroll_into_view_if_needed is a no-op here: the button is
                # already on screen, the figure below it is not.)
                page.evaluate("window.scrollBy(0, 640)")
                settle(page, 1500)
                shot(page, "09b_research_error_map")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! error map failed: {type(exc).__name__}")

        if click_nav(page, "工作台"):
            settle(page, 1200)
            shot(page, "10_dashboard_after_analysis")

        browser.close()
    print(f"\n  screenshots written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
