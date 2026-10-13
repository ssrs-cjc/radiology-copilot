"""Visual design system: colour tokens and the Gradio CSS.

The palette is the one specified for the product.  It is defined once here and
injected as CSS custom properties, so a page never hard-codes a hex value and
the whole app can be re-themed from this file alone.
"""

from __future__ import annotations

import json

# -- brand tokens ----------------------------------------------------------
NAVY = "#143A62"        # primary
BLUE = "#2878D0"        # accent / interactive
TEAL = "#16A6A1"        # secondary accent
BG = "#F5F8FC"          # page background
CARD = "#FFFFFF"
INK = "#243247"         # body text
MUTED = "#66788A"       # secondary text
BORDER = "#E2E9F2"

SUCCESS = "#12805C"
SUCCESS_BG = "#E6F4EF"
WARNING = "#9A6700"
WARNING_BG = "#FFF6E0"
DANGER = "#B42318"
DANGER_BG = "#FEECEB"
INFO_BG = "#EAF2FC"

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", '
    '"Hiragino Sans GB", "Microsoft YaHei", "Source Han Sans SC", '
    '"Noto Sans CJK SC", Roboto, "Helvetica Neue", Arial, sans-serif'
)
MONO_STACK = '"SF Mono", "JetBrains Mono", Menlo, Consolas, "Liberation Mono", monospace'


CSS = f"""
:root {{
  --rc-navy: {NAVY};
  --rc-blue: {BLUE};
  --rc-teal: {TEAL};
  --rc-bg: {BG};
  --rc-card: {CARD};
  --rc-ink: {INK};
  --rc-muted: {MUTED};
  --rc-border: {BORDER};
  --rc-success: {SUCCESS};
  --rc-success-bg: {SUCCESS_BG};
  --rc-warning: {WARNING};
  --rc-warning-bg: {WARNING_BG};
  --rc-danger: {DANGER};
  --rc-danger-bg: {DANGER_BG};
  --rc-info-bg: {INFO_BG};
  --rc-radius: 12px;
  --rc-radius-sm: 8px;
  --rc-shadow: 0 1px 2px rgba(20,58,98,.06), 0 4px 16px rgba(20,58,98,.06);
  --rc-shadow-sm: 0 1px 3px rgba(20,58,98,.08);
}}

/* ---------------------------------------------------------------- base -- */
.gradio-container {{
  background: var(--rc-bg) !important;
  font-family: {FONT_STACK} !important;
  color: var(--rc-ink);
  max-width: 100% !important;
}}
body, .gradio-container, .prose {{ font-family: {FONT_STACK} !important; }}
.gradio-container .prose :is(h1,h2,h3,h4) {{ color: var(--rc-navy); font-weight: 650; }}
footer {{ display: none !important; }}

/* --------------------------------------------------------------- topbar -- */
.rc-topbar {{
  display: flex; align-items: center; gap: 18px;
  background: linear-gradient(96deg, {NAVY} 0%, #1B4A7A 58%, #1D5E86 100%);
  padding: 13px 22px; border-radius: var(--rc-radius);
  box-shadow: var(--rc-shadow); margin-bottom: 14px;
  flex-wrap: wrap;
}}
.rc-brand {{ display: flex; align-items: center; gap: 12px; }}
.rc-logo {{
  width: 38px; height: 38px; border-radius: 10px; flex: 0 0 auto;
  background: linear-gradient(140deg, {TEAL} 0%, {BLUE} 100%);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; color: #fff; font-size: 17px; letter-spacing: .5px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.22);
}}
.rc-title {{ color: #fff; font-size: 17px; font-weight: 650; line-height: 1.2; }}
.rc-subtitle {{ color: rgba(255,255,255,.72); font-size: 11.5px; letter-spacing: .3px; }}
.rc-topbar-spacer {{ flex: 1 1 auto; }}
.rc-topbar-badges {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.rc-badge {{
  font-size: 11.5px; padding: 4px 10px; border-radius: 999px;
  background: rgba(255,255,255,.13); color: #EAF2FC;
  border: 1px solid rgba(255,255,255,.20); white-space: nowrap;
  flex: 0 0 auto; display: inline-flex; align-items: center; gap: 4px;
}}
.rc-badge strong {{ color: #fff; font-weight: 600; }}
.rc-badge--warn {{ background: rgba(255,193,77,.18); border-color: rgba(255,193,77,.4); color: #FFE7B8; }}
.rc-badge--ok {{ background: rgba(22,166,161,.22); border-color: rgba(22,166,161,.45); color: #C9F2EF; }}

/* ---------------------------------------------------------------- cards -- */
.rc-card {{
  background: var(--rc-card); border: 1px solid var(--rc-border);
  border-radius: var(--rc-radius); padding: 16px 18px;
  box-shadow: var(--rc-shadow-sm); margin-bottom: 12px;
}}
.rc-card--flush {{ padding: 0; overflow: hidden; }}
.rc-card-title {{
  font-size: 13px; font-weight: 650; color: var(--rc-navy);
  margin: 0 0 10px 0; display: flex; align-items: center; gap: 7px;
  letter-spacing: .2px;
}}
.rc-card-title .rc-dot {{
  width: 6px; height: 6px; border-radius: 50%; background: var(--rc-teal); flex: 0 0 auto;
}}

/* --------------------------------------------------------------- hero ---- */
.rc-hero {{
  background: var(--rc-card); border: 1px solid var(--rc-border);
  border-radius: var(--rc-radius); padding: 22px 24px; margin-bottom: 14px;
  box-shadow: var(--rc-shadow-sm);
  border-left: 4px solid var(--rc-blue);
}}
.rc-hero h1 {{
  font-size: 25px; margin: 0 0 5px 0; color: var(--rc-navy);
  font-weight: 700; letter-spacing: -.2px;
}}
.rc-hero p {{ margin: 0; color: var(--rc-muted); font-size: 13.5px; line-height: 1.6; }}

/* ------------------------------------------------------- feature cards --- */
.rc-features {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
@media (max-width: 1180px) {{ .rc-features {{ grid-template-columns: repeat(2, 1fr); }} }}
.rc-feature {{
  background: var(--rc-card); border: 1px solid var(--rc-border);
  border-radius: var(--rc-radius); padding: 16px 16px 14px 16px;
  box-shadow: var(--rc-shadow-sm); transition: transform .14s ease, box-shadow .14s ease;
  border-top: 3px solid var(--rc-blue);
}}
.rc-feature:hover {{ transform: translateY(-2px); box-shadow: var(--rc-shadow); }}
.rc-feature:nth-child(2) {{ border-top-color: var(--rc-teal); }}
.rc-feature:nth-child(3) {{ border-top-color: #7A5AF8; }}
.rc-feature:nth-child(4) {{ border-top-color: #E58A2E; }}
.rc-feature-icon {{
  width: 32px; height: 32px; border-radius: 9px; background: var(--rc-info-bg);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; margin-bottom: 10px;
}}
.rc-feature h3 {{ margin: 0 0 6px 0; font-size: 14px; color: var(--rc-navy); font-weight: 650; }}
.rc-feature p {{ margin: 0; font-size: 12.5px; color: var(--rc-muted); line-height: 1.55; }}

/* -------------------------------------------------------------- status --- */
.rc-status-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
@media (max-width: 1180px) {{ .rc-status-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.rc-stat {{
  background: var(--rc-card); border: 1px solid var(--rc-border);
  border-radius: var(--rc-radius-sm); padding: 12px 14px;
}}
.rc-stat-label {{
  font-size: 11.5px; color: var(--rc-muted); margin-bottom: 5px;
  text-transform: uppercase; letter-spacing: .5px; font-weight: 600;
}}
.rc-stat-value {{ font-size: 15px; color: var(--rc-navy); font-weight: 650; line-height: 1.35; }}
.rc-stat-value--empty {{ color: var(--rc-muted); font-weight: 500; font-size: 13px; }}
.rc-stat-sub {{ font-size: 11.5px; color: var(--rc-muted); margin-top: 3px; }}

/* --------------------------------------------------------------- pills --- */
.rc-pill {{
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px;
  border: 1px solid transparent; white-space: nowrap;
}}
.rc-pill--ok {{ background: var(--rc-success-bg); color: var(--rc-success); border-color: #B7E1D2; }}
.rc-pill--warn {{ background: var(--rc-warning-bg); color: var(--rc-warning); border-color: #F2DCA6; }}
.rc-pill--err {{ background: var(--rc-danger-bg); color: var(--rc-danger); border-color: #F5C6C2; }}
.rc-pill--info {{ background: var(--rc-info-bg); color: var(--rc-blue); border-color: #C6DDF6; }}
.rc-pill--muted {{ background: #F1F5F9; color: var(--rc-muted); border-color: #E2E9F2; }}

/* ---------------------------------------------------------------- nav ---- */
.rc-nav .rc-nav-btn button {{
  justify-content: flex-start !important; text-align: left !important;
  background: transparent !important; border: 1px solid transparent !important;
  color: var(--rc-ink) !important; font-weight: 550 !important;
  font-size: 13.5px !important; padding: 9px 13px !important;
  border-radius: var(--rc-radius-sm) !important; box-shadow: none !important;
  min-height: 40px !important; width: 100% !important;
}}
.rc-nav .rc-nav-btn button:hover {{
  background: var(--rc-info-bg) !important; color: var(--rc-navy) !important;
}}
.rc-nav .rc-nav-btn--active button {{
  background: linear-gradient(92deg, {NAVY} 0%, #1D5A8C 100%) !important;
  color: #fff !important; font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(20,58,98,.22) !important;
}}

/* -------------------------------------------------------------- viewer --- */
.rc-viewer .image-container, .rc-viewer img {{
  border-radius: var(--rc-radius-sm) !important;
  background: #0B1220 !important;
}}
.rc-viewer .image-container {{ border: 1px solid var(--rc-border) !important; }}
.rc-view-label {{
  font-size: 11.5px; color: var(--rc-muted); font-weight: 600;
  letter-spacing: .4px; margin-bottom: 6px; text-transform: uppercase;
}}

/* -------------------------------------------------------------- legend --- */
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; padding: 7px 0 2px 0; }}
.legend-item {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--rc-ink); font-weight: 550;
}}
.legend-swatch {{
  width: 13px; height: 13px; border-radius: 3.5px; display: inline-block;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,.14);
}}

/* --------------------------------------------------------- measurement --- */
.rc-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }}
/* The viewer's right rail is narrow; three tiles side by side would clip their
   labels, so there they stack in a single column. */
.rc-metrics--stack {{ grid-template-columns: 1fr; gap: 7px; }}
.rc-metrics--stack .rc-metric {{
  display: grid; grid-template-columns: auto 1fr auto; align-items: center;
  gap: 10px; text-align: left; padding: 8px 11px;
}}
.rc-metrics--stack .rc-metric-region {{ margin: 0; }}
.rc-metrics--stack .rc-metric-value {{ margin: 0; font-size: 16px; }}
.rc-metrics--stack .rc-metric-unit {{ margin: 0; font-size: 10.5px; text-align: right; }}
.rc-metrics--stack .rc-metric-bar {{ display: none; }}
.rc-metric {{
  background: linear-gradient(180deg,#FBFDFF 0%, #F4F8FD 100%);
  border: 1px solid var(--rc-border); border-radius: var(--rc-radius-sm);
  padding: 11px 12px; text-align: center;
}}
.rc-metric-region {{ font-size: 11.5px; color: var(--rc-muted); font-weight: 650; letter-spacing: .6px; }}
.rc-metric-value {{ font-size: 19px; color: var(--rc-navy); font-weight: 700; margin: 3px 0 1px 0; }}
.rc-metric-unit {{ font-size: 11px; color: var(--rc-muted); }}
.rc-metric-bar {{ height: 4px; border-radius: 3px; margin-top: 7px; }}

/* ------------------------------------------------------------- banners --- */
.rc-banner {{
  border-radius: var(--rc-radius-sm); padding: 11px 14px; font-size: 12.8px;
  line-height: 1.6; border: 1px solid; margin-bottom: 10px;
}}
.rc-banner--info {{ background: var(--rc-info-bg); border-color: #C6DDF6; color: #1B4A7A; }}
.rc-banner--warn {{ background: var(--rc-warning-bg); border-color: #F2DCA6; color: #7A5200; }}
.rc-banner--err  {{ background: var(--rc-danger-bg); border-color: #F5C6C2; color: #8F1D14; }}
.rc-banner--ok   {{ background: var(--rc-success-bg); border-color: #B7E1D2; color: #0B6349; }}
.rc-banner strong {{ font-weight: 650; }}

/* ------------------------------------------------------------- trace ----- */
.rc-trace {{
  background: #F8FAFD; border: 1px solid var(--rc-border);
  border-radius: var(--rc-radius-sm); padding: 10px 12px; font-size: 12.5px;
}}
.rc-trace-step {{
  display: flex; gap: 8px; align-items: flex-start;
  padding: 5px 0; border-bottom: 1px dashed #E2E9F2;
}}
.rc-trace-step:last-child {{ border-bottom: none; }}
.rc-trace-tool {{
  font-family: {MONO_STACK}; font-size: 11.5px; color: var(--rc-blue);
  background: var(--rc-info-bg); padding: 1px 6px; border-radius: 5px;
  border: 1px solid #C6DDF6; white-space: nowrap;
}}
.rc-trace-idx {{
  font-size: 10.5px; color: var(--rc-muted); font-weight: 700;
  min-width: 14px; text-align: right;
}}
.rc-trace-args {{
  font-family: {MONO_STACK}; font-size: 10.5px; color: #7A8CA0;
  background: #F1F5F9; padding: 1px 5px; border-radius: 4px;
  max-width: 190px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.rc-trace-summary {{ flex: 1 1 auto; }}
.rc-trace-ms {{
  font-family: {MONO_STACK}; font-size: 10.5px; color: var(--rc-muted);
  margin-left: auto; padding-left: 8px; white-space: nowrap;
}}

/* --------------------------------------------------------------- misc ---- */
.rc-kv {{ display: grid; grid-template-columns: auto 1fr; gap: 5px 12px; font-size: 12.5px; }}
.rc-kv dt {{ color: var(--rc-muted); white-space: nowrap; }}
.rc-kv dd {{ margin: 0; color: var(--rc-ink); font-weight: 550; word-break: break-word; }}

.rc-empty {{
  text-align: center; padding: 26px 18px; color: var(--rc-muted);
  background: #FBFDFF; border: 1px dashed #D3DEEC; border-radius: var(--rc-radius-sm);
  font-size: 13px; line-height: 1.7;
}}
.rc-empty-icon {{ font-size: 26px; display: block; margin-bottom: 8px; opacity: .55; }}

.rc-banner--compact {{ padding: 7px 12px; font-size: 12px; line-height: 1.5; margin-bottom: 8px; }}
.rc-banner-ico {{ display: inline-flex; vertical-align: -3px; margin-right: 7px; }}
.rc-badge svg {{ vertical-align: -2px; margin-right: 4px; }}
.rc-empty-icon svg {{ display: block; margin: 0 auto 8px auto; }}

/* ------------------------------------------------------- case identity --- */
.rc-casecard {{ display: flex; flex-direction: column; gap: 8px; }}
.rc-casecard-id {{
  font-family: {MONO_STACK}; font-size: 13px; font-weight: 650;
  color: var(--rc-navy); line-height: 1.35;
  overflow-wrap: anywhere; word-break: break-word;
}}
.rc-casecard-pills {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.rc-casecard-meta {{
  font-size: 12px; color: var(--rc-muted); line-height: 1.6;
  display: flex; flex-direction: column; gap: 3px;
}}
.rc-casecard-meta span {{
  display: inline-block; min-width: 34px; color: #93A3B5;
  font-size: 11px; margin-right: 6px;
}}

/* --------------------------------------------- bounded scroll regions --- */
/* Chat, tool trace and report preview each scroll on their own so one long
   list cannot push the rest of the page out of view. */
.rc-chat {{ max-height: 186px; overflow-y: auto !important; }}
.rc-trace-scroll {{
  max-height: 258px; overflow-y: auto; border-radius: var(--rc-radius-sm);
  padding-right: 2px;
}}
.rc-trace-scroll::-webkit-scrollbar,
.rc-chat::-webkit-scrollbar,
.rc-report-preview::-webkit-scrollbar {{ width: 8px; }}
.rc-trace-scroll::-webkit-scrollbar-thumb,
.rc-chat::-webkit-scrollbar-thumb,
.rc-report-preview::-webkit-scrollbar-thumb {{
  background: #D3DEEC; border-radius: 4px;
}}
.rc-report-preview {{
  max-height: 470px !important; overflow-y: auto !important; padding: 2px 8px 2px 4px;
  border: 1px solid var(--rc-border); border-radius: var(--rc-radius-sm);
  background: #FCFDFF; font-size: 13px;
}}
/* Gradio nests the rendered markdown in a .prose wrapper; without this the
   inner block grows past the container and the scroll never engages. */
.rc-report-preview .prose {{ max-height: none; overflow: visible; }}
.rc-report-preview h1 {{ font-size: 18px; margin: 10px 0 8px 0; }}
.rc-report-preview h2 {{
  font-size: 14.5px; margin: 16px 0 8px 0; color: var(--rc-navy);
  border-bottom: 1px solid var(--rc-border); padding-bottom: 5px;
}}
.rc-report-preview h3 {{ font-size: 13.5px; margin: 12px 0 6px 0; color: var(--rc-navy); }}
.rc-report-preview table {{
  border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12.5px;
}}
.rc-report-preview th, .rc-report-preview td {{
  border: 1px solid var(--rc-border); padding: 5px 9px; text-align: left;
}}
.rc-report-preview th {{ background: var(--rc-info-bg); color: var(--rc-navy); font-weight: 600; }}
.rc-report-preview blockquote {{
  border-left: 3px solid var(--rc-teal); background: #F6FBFA;
  margin: 8px 0; padding: 8px 12px; color: var(--rc-muted); font-size: 12.5px;
}}
.rc-report-preview ul, .rc-report-preview ol {{ padding-left: 20px; line-height: 1.75; }}
.rc-report-preview code {{
  font-family: {MONO_STACK}; font-size: 11.5px;
  background: #F1F5F9; padding: 1px 5px; border-radius: 4px;
}}

.rc-footer {{
  text-align: center; color: var(--rc-muted); font-size: 11.5px;
  padding: 16px 10px 6px 10px; line-height: 1.7;
  border-top: 1px solid var(--rc-border); margin-top: 16px;
}}
.rc-footer strong {{ color: var(--rc-navy); }}
.rc-hint {{ font-size: 11.8px; color: var(--rc-muted); line-height: 1.6; margin-top: 4px; }}

/* Tighten Gradio's default control chrome. */
.gradio-container .block {{ border: none !important; background: transparent !important; }}
.gradio-container .form {{ background: transparent !important; border: none !important; }}
.gradio-container button.primary {{
  background: var(--rc-blue) !important; border: none !important;
  font-weight: 600 !important; box-shadow: 0 1px 3px rgba(40,120,208,.3) !important;
}}
.gradio-container button.primary:hover {{ background: #1F66B8 !important; }}
.gradio-container button.secondary {{
  background: #fff !important; border: 1px solid var(--rc-border) !important;
  color: var(--rc-ink) !important; font-weight: 550 !important;
}}
.gradio-container button.secondary:hover {{ border-color: var(--rc-blue) !important; color: var(--rc-blue) !important; }}
.gradio-container .tabs {{ border: none !important; }}
.gradio-container label span {{ font-size: 12.5px !important; color: var(--rc-muted) !important; }}
.gradio-container .gr-box, .gradio-container .gr-panel {{ border-color: var(--rc-border) !important; }}
"""


# --------------------------------------------------------------------------
# HTML component helpers
# --------------------------------------------------------------------------
def pill(text: str, kind: str = "info") -> str:
    """Inline status pill. ``kind`` is one of ok/warn/err/info/muted."""
    return f'<span class="rc-pill rc-pill--{kind}">{text}</span>'


# --------------------------------------------------------------------------
# Icons
# --------------------------------------------------------------------------
# Emoji are avoided throughout: many Linux images and headless browsers have no
# emoji font, and the glyphs silently degrade to tofu boxes.  Inline SVG uses
# whatever font/colour the surrounding CSS sets and renders identically
# everywhere.
_ICON_PATHS: dict[str, str] = {
    "brain": "M12 3a3 3 0 0 0-3 3 3 3 0 0 0-3 3 3 3 0 0 0 .5 1.7A3 3 0 0 0 5 13.5 3 3 0 0 0 7 16.4V18a3 3 0 0 0 5 2.2A3 3 0 0 0 17 18v-1.6a3 3 0 0 0 2-2.9 3 3 0 0 0-1.5-2.8A3 3 0 0 0 18 9a3 3 0 0 0-3-3 3 3 0 0 0-3-3z",
    "report": "M7 3h7l5 5v13a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm7 1.5V9h4.5M9 13h6M9 17h4",
    "teaching": "M3 8l9-4 9 4-9 4-9-4zm3 3v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5",
    "research": "M4 20V10m5 10V4m5 16v-7m5 7V8",
    "dashboard": "M4 11l8-7 8 7v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8z",
    "ruler": "M3 8h18v8H3zM7 8v3m4-3v4m4-4v3m4-3v4",
    "target": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm0 5a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0 3a1 1 0 1 0 0 2 1 1 0 0 0 0-2z",
    "folder": "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z",
    "check": "M20 6L9 17l-5-5",
    "cross": "M18 6L6 18M6 6l12 12",
    "warn": "M12 3l9.5 17H2.5L12 3zm0 6v5m0 3v.5",
    "info": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm0 4v.5m0 3V17",
    "lock": "M6 11V8a6 6 0 1 1 12 0v3M5 11h14v10H5z",
    "eye": "M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6zm10 2.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z",
    "download": "M12 3v12m0 0l-4.5-4.5M12 15l4.5-4.5M4 20h16",
    "refresh": "M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5",
    "play": "M7 4l12 8-12 8V4z",
    "microscope": "M6 20h12M9 20V9a2 2 0 0 1 2-2h1a2 2 0 0 1 2 2v11M8 9h7m-3.5 -4V3",
    "sparkle": "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z",
    "database": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
    "upload": "M12 21V9m0 0l-4.5 4.5M12 9l4.5 4.5M4 4h16",
}


def icon(name: str, size: int = 15, color: str = "currentColor", stroke: float = 1.8) -> str:
    """Inline SVG icon. Unknown names fall back to a neutral info glyph."""
    path = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex:0 0 auto;vertical-align:-2px">'
        f'<path d="{path}"/></svg>'
    )


_KIND_ICON = {"ok": "check", "warn": "warn", "err": "cross", "info": "info"}
_KIND_COLOR = {"ok": SUCCESS, "warn": WARNING, "err": DANGER, "info": BLUE}


def card(title: str, body: str, *, dot: bool = True) -> str:
    marker = '<span class="rc-dot"></span>' if dot else ""
    head = f'<div class="rc-card-title">{marker}{title}</div>' if title else ""
    return f'<div class="rc-card">{head}{body}</div>'


def banner(text: str, kind: str = "info", compact: bool = False) -> str:
    glyph = icon(_KIND_ICON.get(kind, "info"), 15, _KIND_COLOR.get(kind, BLUE))
    extra = " rc-banner--compact" if compact else ""
    return (f'<div class="rc-banner rc-banner--{kind}{extra}">'
            f'<span class="rc-banner-ico">{glyph}</span>{text}</div>')


def empty_state(icon_name: str, title: str, hint: str = "") -> str:
    hint_html = f"<br><span style='font-size:12px'>{hint}</span>" if hint else ""
    glyph = icon(icon_name, 26, "#9FB2C8", stroke=1.5)
    return (
        f'<div class="rc-empty"><span class="rc-empty-icon">{glyph}</span>'
        f"<strong>{title}</strong>{hint_html}</div>"
    )


def stat(label: str, value: str | None, sub: str = "") -> str:
    if value is None or value == "":
        body = '<div class="rc-stat-value rc-stat-value--empty">暂无数据</div>'
    else:
        body = f'<div class="rc-stat-value">{value}</div>'
    sub_html = f'<div class="rc-stat-sub">{sub}</div>' if sub else ""
    return f'<div class="rc-stat"><div class="rc-stat-label">{label}</div>{body}{sub_html}</div>'


def kv(pairs: list[tuple[str, str]]) -> str:
    rows = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs)
    return f'<dl class="rc-kv">{rows}</dl>'


def metric_tile(region: str, region_zh: str, value_ml: float | None, color: str,
                voxel_count: int | None = None) -> str:
    if value_ml is None:
        val, unit = "—", "尚未计算"
    else:
        val, unit = f"{value_ml:.2f}", "mL"
    count = f"{voxel_count:,} 体素" if voxel_count is not None else ""
    return (
        f'<div class="rc-metric">'
        f'<div class="rc-metric-region">{region} · {region_zh}</div>'
        f'<div class="rc-metric-value">{val}</div>'
        f'<div class="rc-metric-unit">{unit}{(" · " + count) if count else ""}</div>'
        f'<div class="rc-metric-bar" style="background:{color}"></div>'
        f"</div>"
    )


def trace_html(steps: list[dict]) -> str:
    if not steps:
        return '<div class="rc-trace"><em>本次未调用任何工具。</em></div>'
    status_icon = {"ok": ("check", SUCCESS), "error": ("warn", WARNING),
                   "refused": ("cross", DANGER)}
    rows = []
    for i, s in enumerate(steps):
        if s.get("kind") != "tool_call":
            continue
        name, colour = status_icon.get(s.get("status", ""), ("info", MUTED))
        glyph = icon(name, 14, colour)
        ms = s.get("duration_ms")
        timing = (f'<span class="rc-trace-ms">{ms:.0f} ms</span>'
                  if isinstance(ms, (int, float)) else "")
        args = s.get("arguments")
        args_html = ""
        if args:
            try:
                args_html = (f'<span class="rc-trace-args">'
                             f'{json.dumps(args, ensure_ascii=False)}</span>')
            except (TypeError, ValueError):
                args_html = ""
        err = (f'<div style="color:{DANGER};font-size:11.5px">{s["error"]}</div>'
               if s.get("error") else "")
        rows.append(
            f'<div class="rc-trace-step">'
            f'<span class="rc-trace-idx">{i + 1}</span>'
            f'<span>{glyph}</span>'
            f'<span class="rc-trace-tool">{s.get("tool","")}</span>'
            f'{args_html}'
            f'<span class="rc-trace-summary">{s.get("summary","")}{err}</span>'
            f'{timing}</div>'
        )
    return f'<div class="rc-trace">{"".join(rows)}</div>'


def topbar(case_id: str | None, model_mode: str, model_kind: str,
           origin_label: str | None = None) -> str:
    badges = []
    if case_id:
        badges.append(f'<span class="rc-badge">当前病例 <strong>{case_id}</strong></span>')
    else:
        badges.append('<span class="rc-badge">当前病例 <strong>未选择</strong></span>')
    badges.append(f'<span class="rc-badge {model_kind}">模型 <strong>{model_mode}</strong></span>')
    if origin_label:
        badges.append(f'<span class="rc-badge">{origin_label}</span>')
    badges.append(
        '<span class="rc-badge rc-badge--warn">'
        + icon("warn", 13, "#FFD79A")
        + " 科研教学用途 · 非临床诊断</span>"
    )
    return (
        '<div class="rc-topbar">'
        '<div class="rc-brand">'
        '<div class="rc-logo">RC</div>'
        "<div>"
        '<div class="rc-title">Radiology Copilot</div>'
        '<div class="rc-subtitle">AI-Powered Radiology · 影像智能诊教研工作台</div>'
        "</div></div>"
        '<div class="rc-topbar-spacer"></div>'
        f'<div class="rc-topbar-badges">{"".join(badges)}</div>'
        "</div>"
    )
