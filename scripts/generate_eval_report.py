"""
generate_eval_report.py
-----------------------
Generate a self-contained HTML visual evaluation report for human annotation review.

Opens in any browser – no server required.
Each sample card shows:
  - CAM_FRONT keyframe image  (+ optional 2-frame history strip)
  - Driving decision badges (longitudinal / lateral)
  - VLM cot reasoning
  - Critical components breakdown
  - Rating buttons (✓ Correct / ~ Partial / ✗ Wrong)  – saved in browser localStorage
  - Preliminary rule-based reasoning for comparison

Usage:
    python scripts/generate_eval_report.py \
        --input out/nuscenes_coc_v2_final_teacher.json \
        --output out/eval_report.html
"""

import argparse
import base64
import json
import os
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def img_to_b64(path: str) -> str:
    """Return a base64-encoded data URI for an image file, or empty string."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lstrip(".").lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return ""


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


DECISION_COLORS = {
    # longitudinal
    "lead_obstacle_following":  "#2563eb",
    "speed_adaptation_road":    "#7c3aed",
    "stop_static_constraint":   "#dc2626",
    "yield_agent_right_of_way": "#b45309",
    "maintain_speed":           "#16a34a",
    "set_speed_tracking":       "#0891b2",
    # lateral
    "lane_keeping_centering":   "#6b7280",
    "in_lane_nudge_left":       "#d97706",
    "in_lane_nudge_right":      "#d97706",
    "lane_change_left":         "#db2777",
    "lane_change_right":        "#db2777",
    "none":                     "#9ca3af",
}


def badge(label: str) -> str:
    color = DECISION_COLORS.get(label, "#6b7280")
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:12px;font-weight:600;margin-right:4px">'
            f'{html_escape(label)}</span>')


def render_components(components: list) -> str:
    if not components:
        return '<em style="color:#9ca3af">none</em>'
    rows = []
    for c in components:
        cat = c.get("category", "")
        attrs = c.get("attributes", {})
        conf = c.get("confidence", "")
        cat_colors = {
            "critical_objects": "#1d4ed8",
            "road_events":      "#7c3aed",
            "traffic_controls": "#dc2626",
            "lane_info":        "#16a34a",
            "ego_motion":       "#0891b2",
        }
        cat_color = cat_colors.get(cat, "#6b7280")
        attr_str = html_escape(json.dumps(attrs, ensure_ascii=False))
        rows.append(
            f'<tr>'
            f'<td style="padding:4px 8px;white-space:nowrap">'
            f'<span style="color:{cat_color};font-weight:600;font-size:12px">{html_escape(cat)}</span>'
            f'</td>'
            f'<td style="padding:4px 8px;font-size:12px;font-family:monospace;color:#374151">'
            f'{attr_str}'
            f'</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:#9ca3af">{html_escape(conf)}</td>'
            f'</tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;background:#f9fafb;border-radius:6px">'
        '<thead><tr>'
        '<th style="padding:4px 8px;text-align:left;font-size:11px;color:#6b7280">category</th>'
        '<th style="padding:4px 8px;text-align:left;font-size:11px;color:#6b7280">attributes</th>'
        '<th style="padding:4px 8px;text-align:left;font-size:11px;color:#6b7280">conf</th>'
        '</tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
    )


def render_sample(idx: int, sample: dict) -> str:
    sid      = html_escape(sample.get("sample_id", ""))
    token    = html_escape(sample.get("nuscenes_sample_token", ""))
    dd       = sample.get("driving_decision", {})
    lon      = dd.get("longitudinal", "none")
    lat      = dd.get("lateral", "none")
    cot      = html_escape(sample.get("coc_reasoning", ""))
    prelim   = html_escape(sample.get("preliminary_coc", {}).get("reasoning", ""))
    src      = html_escape(sample.get("teacher_source", ""))
    conf     = sample.get("teacher_confidence", "")
    evid     = html_escape(str(sample.get("teacher_evidence", "")))
    components = sample.get("critical_components", [])

    # images: keyframe + up to 2 history frames
    img_paths = []
    hist = sample.get("history_frames", [])
    for hf in sorted(hist, key=lambda x: x.get("relative_index", 0)):
        p = hf.get("camera_paths", {}).get("CAM_FRONT", "")
        if p:
            img_paths.append(("history " + str(hf.get("relative_index", "")), p))
    kp = sample.get("camera_paths", {}).get("CAM_FRONT", "")
    if kp:
        img_paths.append(("keyframe ★", kp))

    img_html = ""
    for label, path in img_paths:
        b64 = img_to_b64(path)
        if b64:
            border = "3px solid #2563eb" if "keyframe" in label else "1px solid #e5e7eb"
            img_html += (
                f'<div style="text-align:center;margin-right:6px">'
                f'<div style="font-size:10px;color:#9ca3af;margin-bottom:2px">{html_escape(label)}</div>'
                f'<img src="{b64}" style="height:160px;border-radius:6px;border:{border}" />'
                f'</div>'
            )
        else:
            img_html += f'<div style="width:240px;height:160px;background:#f3f4f6;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px">no image</div>'

    conf_color = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}.get(
        str(conf).lower(), "#6b7280"
    )

    return f"""
<div class="sample-card" id="card-{idx}"
     style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:20px;margin-bottom:28px;box-shadow:0 1px 4px rgba(0,0,0,.07)">

  <!-- header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <div>
      <span style="font-size:13px;font-weight:700;color:#111827">#{idx+1}</span>
      <span style="font-size:11px;color:#9ca3af;margin-left:8px">{sid}</span>
    </div>
    <div>
      {badge(lon)}
      {badge(lat)}
      <span style="font-size:11px;color:{conf_color};margin-left:6px">conf:{conf}</span>
      <span style="font-size:11px;color:#9ca3af;margin-left:6px">src:{src}</span>
    </div>
  </div>

  <!-- images -->
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;overflow-x:auto">
    {img_html}
  </div>

  <!-- cot reasoning -->
  <div style="margin-bottom:12px">
    <div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em">VLM Chain-of-Thought</div>
    <div style="background:#eff6ff;border-left:3px solid #2563eb;padding:10px 14px;border-radius:0 6px 6px 0;font-size:14px;color:#1e3a8a;line-height:1.6">
      {cot}
    </div>
  </div>

  <!-- rule-based preliminary -->
  <div style="margin-bottom:12px">
    <div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em">Rule-based Preliminary</div>
    <div style="background:#f9fafb;border-left:3px solid #d1d5db;padding:8px 14px;border-radius:0 6px 6px 0;font-size:13px;color:#6b7280;line-height:1.5">
      {prelim if prelim else '<em>n/a</em>'}
    </div>
  </div>

  <!-- evidence -->
  <div style="margin-bottom:12px">
    <div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em">Evidence</div>
    <div style="font-size:12px;color:#374151;font-family:monospace;background:#f9fafb;padding:6px 10px;border-radius:6px">{evid}</div>
  </div>

  <!-- critical components -->
  <div style="margin-bottom:16px">
    <div style="font-size:11px;font-weight:600;color:#6b7280;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">Critical Components</div>
    {render_components(components)}
  </div>

  <!-- rating buttons -->
  <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
    <span style="font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase">Your Rating:</span>
    <button onclick="rate({idx},'correct')"   class="btn-rate" id="r-{idx}-correct"  style="background:#16a34a">✓ Correct</button>
    <button onclick="rate({idx},'partial')"   class="btn-rate" id="r-{idx}-partial"  style="background:#d97706">~ Partial</button>
    <button onclick="rate({idx},'wrong')"     class="btn-rate" id="r-{idx}-wrong"    style="background:#dc2626">✗ Wrong</button>
    <button onclick="rate({idx},'skip')"      class="btn-rate" id="r-{idx}-skip"     style="background:#9ca3af">– Skip</button>
    <span id="note-{idx}" style="font-size:12px;color:#6b7280;margin-left:8px"></span>
  </div>
  <textarea id="comment-{idx}" placeholder="Optional comment..." onchange="saveComment({idx})"
    style="margin-top:8px;width:100%;box-sizing:border-box;height:50px;font-size:12px;border:1px solid #e5e7eb;border-radius:6px;padding:6px;color:#374151;resize:vertical"></textarea>
</div>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate HTML visual evaluation report")
    ap.add_argument("--input",  default="out/nuscenes_coc_v2_final_teacher.json")
    ap.add_argument("--output", default="out/eval_report.html")
    args = ap.parse_args()

    print(f"[INFO] Loading {args.input} ...")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] {len(data)} samples")

    cards_html = "\n".join(render_sample(i, s) for i, s in enumerate(data))

    # Decision distribution for summary bar
    from collections import Counter
    lon_dist = Counter(s.get("driving_decision", {}).get("longitudinal", "none") for s in data)
    lat_dist = Counter(s.get("driving_decision", {}).get("lateral", "none") for s in data)

    dist_rows = ""
    for k, v in sorted(lon_dist.items(), key=lambda x: -x[1]):
        color = DECISION_COLORS.get(k, "#6b7280")
        dist_rows += f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px"><span style="width:200px;font-size:12px;color:#374151">{html_escape(k)}</span><div style="background:{color};height:14px;width:{v*30}px;border-radius:3px"></div><span style="font-size:12px;color:#6b7280">{v}</span></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoC Annotation Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f3f4f6; margin:0; padding:20px; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  .btn-rate {{
    border:none; color:#fff; padding:6px 14px; border-radius:6px; font-size:13px; font-weight:600;
    cursor:pointer; opacity:0.55; transition:opacity .15s, box-shadow .15s;
  }}
  .btn-rate:hover {{ opacity:0.85; }}
  .btn-rate.active {{ opacity:1; box-shadow:0 0 0 3px rgba(0,0,0,.25); }}
  #summary-bar {{ background:#fff; border-radius:10px; padding:16px 20px; margin-bottom:24px; box-shadow:0 1px 4px rgba(0,0,0,.07); }}
  #progress {{ font-size:14px; color:#374151; margin-bottom:8px; }}
</style>
</head>
<body>
<div class="container">

  <h1 style="font-size:24px;font-weight:700;color:#111827;margin-bottom:4px">CoC Annotation Evaluation</h1>
  <p style="color:#6b7280;font-size:14px;margin-bottom:20px">
    {len(data)} samples &nbsp;|&nbsp; Rate each annotation then export your results.
  </p>

  <!-- summary bar -->
  <div id="summary-bar">
    <div id="progress">Rated: <b id="rated-count">0</b> / {len(data)} &nbsp;&nbsp; ✓ <b id="c-correct" style="color:#16a34a">0</b> &nbsp; ~ <b id="c-partial" style="color:#d97706">0</b> &nbsp; ✗ <b id="c-wrong" style="color:#dc2626">0</b></div>
    <button onclick="exportRatings()" style="background:#2563eb;color:#fff;border:none;padding:6px 16px;border-radius:6px;font-size:13px;cursor:pointer;font-weight:600">Export Ratings JSON</button>

    <details style="margin-top:12px">
      <summary style="cursor:pointer;font-size:13px;font-weight:600;color:#6b7280">Decision Distribution</summary>
      <div style="margin-top:10px">{dist_rows}</div>
    </details>
  </div>

  <!-- sample cards -->
  {cards_html}

</div>
<script>
const RATINGS = {{}};
const COMMENTS = {{}};
const TOTAL = {len(data)};

function rate(idx, value) {{
  RATINGS[idx] = value;
  // update button states
  ['correct','partial','wrong','skip'].forEach(v => {{
    const btn = document.getElementById('r-'+idx+'-'+v);
    btn.classList.toggle('active', v === value);
  }});
  document.getElementById('note-'+idx).textContent = '✔ saved';
  updateSummary();
  localStorage.setItem('coc_eval_ratings', JSON.stringify(RATINGS));
}}

function saveComment(idx) {{
  COMMENTS[idx] = document.getElementById('comment-'+idx).value;
  localStorage.setItem('coc_eval_comments', JSON.stringify(COMMENTS));
}}

function updateSummary() {{
  const vals = Object.values(RATINGS);
  document.getElementById('rated-count').textContent = vals.length;
  document.getElementById('c-correct').textContent = vals.filter(v=>v==='correct').length;
  document.getElementById('c-partial').textContent = vals.filter(v=>v==='partial').length;
  document.getElementById('c-wrong').textContent   = vals.filter(v=>v==='wrong').length;
}}

function exportRatings() {{
  const out = [];
  for (let i = 0; i < TOTAL; i++) {{
    out.push({{ index: i, rating: RATINGS[i] || null, comment: COMMENTS[i] || '' }});
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'coc_eval_ratings.json';
  a.click();
}}

// restore from localStorage on load
(function() {{
  try {{
    const saved = localStorage.getItem('coc_eval_ratings');
    if (saved) {{
      const r = JSON.parse(saved);
      Object.entries(r).forEach(([idx, val]) => {{ if (val) rate(parseInt(idx), val); }});
    }}
    const sc = localStorage.getItem('coc_eval_comments');
    if (sc) {{
      const c = JSON.parse(sc);
      Object.entries(c).forEach(([idx, val]) => {{
        const el = document.getElementById('comment-'+idx);
        if (el) {{ el.value = val; COMMENTS[idx] = val; }}
      }});
    }}
  }} catch(e) {{}}
}})();
</script>
</body>
</html>
"""

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"[INFO] Report saved -> {out_path}  ({size_kb} KB)")
    print(f"[INFO] Open in browser: {out_path.resolve()}")


if __name__ == "__main__":
    main()
