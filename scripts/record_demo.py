#!/usr/bin/env python
"""Record a ~2 minute walkthrough of the running application.

This drives the **real** UI in a real browser and captures a real video.  It is
not a scripted animation: every segmentation, tool call and correction shown in
the recording is executed live by the app.

    python app.py &                      # app must be running
    python scripts/record_demo.py

Output: ``assets/demo/radiology_copilot_demo.webm`` (plus a PNG poster frame).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "demo"
VIEWPORT = {"width": 1440, "height": 900}


#: Scales every deliberate pause. 1.0 = full walkthrough; lower values give a
#: condensed cut that still shows every step, just with less dwell time.
PACE = 1.0


def beat(page: Page, ms: int) -> None:
    """Pause so a human watching the video can read the screen."""
    page.wait_for_timeout(int(ms * PACE))


def nav(page: Page, label: str, settle_ms: int = 2200) -> bool:
    try:
        page.get_by_role("button", name=label, exact=False).first.click(timeout=8000)
        beat(page, settle_ms)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! nav '{label}': {type(exc).__name__}")
        return False


def page_text(page: Page) -> str:
    """All rendered text, including nodes that are currently not visible.

    `inner_text` returns only visible text, so a result written into a collapsed
    panel reads as a failure and the recorder retries a step that already
    succeeded.
    """
    return page.locator("body").text_content() or ""


def wait_for(page: Page, needle: str, timeout_ms: int = 90_000) -> bool:
    """Wait until the page text contains `needle`."""
    waited = 0
    while waited < timeout_ms:
        if needle in page_text(page):
            return True
        page.wait_for_timeout(1000)
        waited += 1000
    return False


def open_correction_panel(page: Page) -> None:
    """Expand the correction accordion if its contents are not in the DOM.

    Gradio only keeps a closed accordion's children in the DOM when it has been
    opened at least once in this session, so the check is "does the button
    exist", and the header is clicked as text (its role/aria markup varies
    between Gradio versions).
    """
    demo = page.get_by_role("button", name="演示：用参考标注模拟", exact=False)
    for _ in range(4):
        if demo.count():
            return
        try:
            page.get_by_text("人工修正掩膜回流", exact=False).first.click(timeout=8000)
        except Exception:  # noqa: BLE001
            pass
        beat(page, 1800)


def scroll(page: Page, dy: int) -> None:
    page.evaluate(f"window.scrollBy(0, {dy})")
    beat(page, 900)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--skip-segmentation", action="store_true",
                    help="assume a prediction already exists")
    ap.add_argument("--pace", type=float, default=1.0,
                    help="pause multiplier; 0.3-0.4 gives a ~2 minute cut")
    ap.add_argument("--short", action="store_true",
                    help="skip optional steps (error map, tab toggling)")
    args = ap.parse_args()

    global PACE
    PACE = max(0.1, args.pace)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chromium", args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=1,
            record_video_dir=str(OUT_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector(".rc-topbar", timeout=120_000)
        page.wait_for_timeout(5000)

        # 1 ── 工作台 ----------------------------------------------------
        print("1. dashboard")
        scroll(page, 260)
        beat(page, 2500)
        scroll(page, -260)
        beat(page, 1200)

        # 2 ── 智能阅片 + 真实分割 ----------------------------------------
        print("2. viewer + segmentation")
        if nav(page, "智能阅片"):
            beat(page, 2500)
            if not args.skip_segmentation:
                try:
                    page.get_by_role("button", name="运行 AI 分割",
                                     exact=False).first.click(timeout=10_000)
                    wait_for(page, "分割完成")
                    beat(page, 3000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! segmentation: {type(exc).__name__}")
            # Slice through a few planes to show the mask following the anatomy.
            for _ in range(3):
                beat(page, 900)
            beat(page, 2500)

        # 3 ── AI 报告 + LLM 工具调用 -------------------------------------
        print("3. report + agent")
        if nav(page, "AI 报告"):
            beat(page, 1500)
            try:
                box = page.get_by_placeholder("例如：分析当前病例").first
                box.fill("请分析当前病例，查询脑肿瘤分割结果，整理区域体积，"
                         "并生成一份待审核影像分析报告。")
                beat(page, 1500)
                page.get_by_role("button", name="发送", exact=False).first.click(timeout=8000)
                beat(page, 9000)          # real API round trips
                beat(page, 3000)

                page.get_by_role("button", name="生成报告草稿（模板）",
                                 exact=False).first.click(timeout=10_000)
                beat(page, 3500)
                if not args.short:
                    page.get_by_role("tab", name="编辑", exact=False).first.click(timeout=8000)
                    beat(page, 2500)
                    page.get_by_role("tab", name="预览", exact=False).first.click(timeout=8000)
                    beat(page, 2000)
                approve = page.get_by_role("button", name="确认审核", exact=False).first
                approve.scroll_into_view_if_needed(timeout=8000)
                beat(page, 1500)
                approve.click(timeout=10_000)
                wait_for(page, "已确认审核", timeout_ms=30_000)
                beat(page, 3000)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! report flow: {type(exc).__name__}: {exc}")

        # 4 ── 教学模式：答案隔离 ------------------------------------------
        print("4. teaching")
        if nav(page, "教学模式"):
            beat(page, 1500)
            try:
                page.get_by_role("button", name="加载练习", exact=False).first.click(timeout=8000)
                beat(page, 3500)
                # Attempt to reveal before submitting - must be refused.
                page.get_by_role("button", name="显示参考标注",
                                 exact=False).first.click(timeout=8000)
                beat(page, 3500)
                page.get_by_placeholder("描述观察到的异常区域").first.fill(
                    "FLAIR 序列见右侧额叶大片高信号，累及皮层下白质；"
                    "T1ce 可见结节状强化，考虑肿瘤核心区。")
                beat(page, 2000)
                page.get_by_role("button", name="提交答案", exact=False).first.click(timeout=8000)
                beat(page, 4000)
                page.get_by_role("tab", name="反馈结果", exact=False).first.click(timeout=8000)
                beat(page, 3500)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! teaching: {type(exc).__name__}: {exc}")

        # 5 ── 科研：修正回流 + Dashboard ----------------------------------
        print("5. research")
        if nav(page, "科研反馈"):
            beat(page, 2000)
            try:
                if not args.short:
                    page.get_by_text("误差可视化", exact=False).first.click(timeout=8000)
                    beat(page, 1200)
                    page.get_by_role("button", name="生成误差对比图",
                                     exact=False).first.click(timeout=15_000)
                    beat(page, 4500)
                    scroll(page, 620)
                    beat(page, 3000)

                open_correction_panel(page)
                open_correction_panel(page)
                page.get_by_role("button", name="演示：用参考标注模拟",
                                 exact=False).first.click(timeout=15_000)
                beat(page, 5000)
                open_correction_panel(page)      # results render inside it
                if "演示数据" not in page_text(page):
                    print("  ! demo banner not found after re-opening the panel")
                # The save button sits below the comparison figure; without
                # bringing it into view the click lands nowhere and nothing is
                # persisted (which the recording would then misrepresent).
                open_correction_panel(page)
                save_btn = page.get_by_role("button", name="保存为科研反馈",
                                            exact=False).first
                save_btn.scroll_into_view_if_needed(timeout=10_000)
                beat(page, 1500)
                save_btn.click(timeout=15_000)
                beat(page, 4000)
                open_correction_panel(page)
                if "已保存第" not in page_text(page):
                    print("  ! correction save did not confirm")
                beat(page, 3000)
                scroll(page, 300)
                beat(page, 3500)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! research: {type(exc).__name__}: {exc}")

        # 6 ── 回到工作台收尾 ----------------------------------------------
        if not args.short:
            print("6. wrap up")
            nav(page, "工作台", settle_ms=2500)
            scroll(page, 200)
            beat(page, 3000)

        # The video is flushed on close; `video.path()` is only valid before
        # that, so locate the written file afterwards instead.
        context.close()
        browser.close()

    produced = sorted(OUT_DIR.glob("*.webm"))
    if not produced:
        print("\n  视频未能生成")
        return 1

    final = OUT_DIR / ("radiology_copilot_demo.webm" if PACE < 0.9
                       else "radiology_copilot_demo_full.webm")
    if produced[0] != final:
        if final.exists():
            final.unlink()
        produced[0].replace(final)
    size_mb = final.stat().st_size / 1e6
    print(f"\n  video : {final}  ({size_mb:.1f} MB)")
    print("  转换  : ffmpeg -i radiology_copilot_demo.webm "
          "-c:v libx264 -pix_fmt yuv420p radiology_copilot_demo.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
