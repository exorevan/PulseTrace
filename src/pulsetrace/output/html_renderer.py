from __future__ import annotations

import re
from datetime import datetime
from itertools import groupby
from pathlib import Path

from pulsetrace.explainers.result import ExplanationResult, FeatureContribution

_SCALE = 0.5  # weight at which bar reaches 100% width


def render(result: ExplanationResult) -> None:
    """Write an ExplanationResult as a self-contained HTML file under outputs/."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"{result.method}_{result.mode}_{result.task}_{timestamp}.html"
    filename.write_text(_build_html(result), encoding="utf-8")
    print(f"  {result.method.upper()} | mode={result.mode} | task={result.task}")
    print(f"HTML saved -> {filename}")


def _param_name(feature: str) -> str:
    m = re.search(r"[A-Za-z_]\w*", feature)
    return m.group(0) if m else feature


def _condition_lower_bound(feature: str) -> float:
    m = re.match(r"^([\d.]+)\s*<", feature)
    if m:
        return float(m.group(1))
    m = re.search(r">\s*([\d.]+)", feature)
    if m:
        return float(m.group(1))
    return float("-inf")


def _bar_html(weight: float) -> str:
    pct = min(abs(weight) / _SCALE, 1.0) * 100
    color = "#2ecc71" if weight >= 0 else "#e74c3c"
    direction = "left" if weight >= 0 else "right"
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};float:{direction}"></div>'
        f"</div>"
    )


def _importance_label(weight: float) -> str:
    if abs(weight) < _SCALE * 0.15:
        return "not important"
    return "raises result" if weight >= 0 else "lowers result"


def _rows_html(contributions: list[FeatureContribution]) -> str:
    rows = []
    sorted_c = sorted(
        contributions,
        key=lambda c: (_param_name(c.feature), _condition_lower_bound(c.feature)),
    )
    prev_param: str | None = None
    for fc in sorted_c:
        current_param = _param_name(fc.feature)
        separator = (
            '<tr class="separator"><td colspan="4"></td></tr>'
            if prev_param is not None and current_param != prev_param
            else ""
        )
        prev_param = current_param
        sign_class = "pos" if fc.weight >= 0 else "neg"
        label = _importance_label(fc.weight)
        rows.append(
            f"{separator}"
            f"<tr>"
            f'<td class="feat">{fc.feature}</td>'
            f'<td class="weight {sign_class}">{fc.weight:+.4f}</td>'
            f'<td class="bar-cell">{_bar_html(fc.weight)}</td>'
            f'<td class="label">{label}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _section_html(label: str | int | None, contributions: list[FeatureContribution], base: float | None) -> str:
    base_str = f'<p class="base">Base value: <strong>{base:+.4f}</strong></p>' if base is not None else ""
    heading = f'<h2 class="class-heading">Class: {label}</h2>' if label is not None else ""
    return f"""
    {heading}
    {base_str}
    <table>
      <thead>
        <tr><th>Feature</th><th>Weight</th><th>Impact</th><th>Interpretation</th></tr>
      </thead>
      <tbody>
        {_rows_html(contributions)}
      </tbody>
    </table>
"""


def _panels_html(panels: list) -> str:
    """Render a list of ImagePanel objects as a responsive image gallery."""
    items = []
    for p in panels:
        conf_str = (
            f'<span class="conf">{p.confidence:.2%}</span>' if p.confidence is not None else ""
        )
        items.append(
            f'<div class="panel">'
            f'<div class="panel-title">{p.title}{conf_str}</div>'
            f'<div class="img-pair">'
            f'<div class="img-wrap"><div class="img-label">Original</div>'
            f'<img class="xai-img" src="data:image/png;base64,{p.original_b64}"></div>'
            f'<div class="img-wrap"><div class="img-label">Explanation</div>'
            f'<img class="xai-img" src="data:image/png;base64,{p.explanation_b64}"></div>'
            f'</div>'
            f'</div>'
        )
    legend = (
        '<div class="legend">'
        '<span class="legend-pos">&#9632; raises result</span>'
        '<span class="legend-neg">&#9632; lowers result</span>'
        '</div>'
    )
    return legend + '<div class="panel-grid">' + "\n".join(items) + "</div>"


def _base_meta(base_values: dict | None) -> str:
    if not base_values:
        return ""
    parts = []
    for k, v in base_values.items():
        label = "base" if k is None else f"base[{k}]"
        parts.append(f'<span class="meta-item">{label}: {v:+.4f}</span>')
    return " ".join(parts)


def _build_html(result: ExplanationResult) -> str:
    samples_line = (
        f'<span class="meta-item">samples: {result.global_samples}</span>'
        if result.global_samples is not None
        else ""
    )
    base_line = _base_meta(result.base_values)

    if result.image_panels:
        body = _panels_html(result.image_panels)
    elif result.task == "regression":
        base = result.base_values.get(None) if result.base_values else None
        body = _section_html(None, result.contributions, base)
    else:
        sorted_c = sorted(result.contributions, key=lambda c: str(c.label))
        groups = [
            (label, list(group))
            for label, group in groupby(sorted_c, key=lambda c: c.label)
        ]
        parts = []
        for label, group in groups:
            base = result.base_values.get(label) if result.base_values else None
            parts.append(_section_html(label, group, base))
        body = "\n".join(parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PulseTrace &mdash; {result.method.upper()} explanation</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #f4f6f9;
      margin: 0;
      padding: 24px;
      color: #2c3e50;
    }}
    header {{
      background: #2c3e50;
      color: #fff;
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 1.4rem; }}
    .meta {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.85rem; opacity: 0.85; }}
    .meta-item {{ background: rgba(255,255,255,0.15); padding: 2px 8px; border-radius: 4px; }}
    .section {{
      background: #fff;
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    h2.class-heading {{ margin: 0 0 12px; font-size: 1.1rem; color: #2c3e50; }}
    p.base {{ margin: 0 0 12px; font-size: 0.9rem; color: #555; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    thead th {{
      text-align: left;
      padding: 6px 10px;
      background: #f0f2f5;
      border-bottom: 2px solid #dde1e7;
      font-weight: 600;
      color: #555;
    }}
    tbody tr:hover {{ background: #fafbfc; }}
    tbody td {{ padding: 5px 10px; border-bottom: 1px solid #eef0f3; vertical-align: middle; }}
    tr.separator td {{ padding: 4px 0; border: none; }}
    td.feat {{ font-family: monospace; font-size: 0.85rem; max-width: 280px; word-break: break-word; }}
    td.weight {{ font-family: monospace; font-weight: 600; width: 80px; }}
    td.weight.pos {{ color: #27ae60; }}
    td.weight.neg {{ color: #c0392b; }}
    td.bar-cell {{ width: 160px; padding: 5px 10px; }}
    td.label {{ font-size: 0.82rem; color: #777; white-space: nowrap; }}
    .bar-track {{
      height: 12px;
      background: #eef0f3;
      border-radius: 6px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 6px;
      transition: width 0.2s;
    }}
    .legend {{ display: flex; gap: 16px; margin-bottom: 14px; font-size: 0.85rem; }}
    .legend-pos {{ color: #27ae60; font-weight: 600; }}
    .legend-neg {{ color: #c0392b; font-weight: 600; }}
    .panel-grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
    .panel {{
      background: #fff;
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .panel-title {{
      font-weight: 600;
      font-size: 0.9rem;
      margin-bottom: 10px;
      color: #2c3e50;
    }}
    .conf {{ font-weight: 400; font-size: 0.82rem; color: #27ae60; margin-left: 8px; }}
    .img-pair {{ display: flex; gap: 10px; }}
    .img-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
    .img-label {{ font-size: 0.75rem; color: #888; }}
    .xai-img {{
      max-width: 320px;
      max-height: 200px;
      width: auto;
      height: auto;
      object-fit: contain;
      border-radius: 4px;
      image-rendering: auto;
    }}
  </style>
</head>
<body>
  <header>
    <h1>PulseTrace &mdash; {result.method.upper()} | {result.mode} | {result.task}</h1>
    <div class="meta">
      <span class="meta-item">target: {result.target_name}</span>
      {samples_line}
      {base_line}
    </div>
  </header>
  <div class="section">
    {body}
  </div>
</body>
</html>"""
