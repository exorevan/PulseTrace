"""Utilities for rendering XAI time series explanations as base64-encoded PNG panels."""
from __future__ import annotations

import base64
import io
import re

import numpy as np

from .result import ImagePanel


def expand_segment_weights(pairs: list[tuple[str, float]], T: int) -> np.ndarray:
    """Expand LIME segment weights to a per-timestep array.

    Segment names follow the pattern 'seg_i [tSTART-tEND]'. Each timestep in
    [START, END] (inclusive) receives the segment's weight. Unmentioned
    timesteps stay 0.0 (near-zero → transparent in the chart).
    """
    weights = np.zeros(T)
    for name, w in pairs:
        m = re.search(r"\[t(\d+)-t(\d+)\]", name)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            weights[start : end + 1] = w
    return weights


def ts_panel(
    series: np.ndarray,
    weights: np.ndarray,
    title: str,
    confidence: float | None = None,
) -> ImagePanel:
    """Build an ImagePanel showing a time series with contribution-colored points.

    Args:
        series: 1-D array of shape (T,) — the raw time series values.
        weights: 1-D array of shape (T,) — per-timestep contribution weights.
            Positive → green, negative → red; magnitude drives opacity/size.
        title: Panel heading shown in the HTML output.
        confidence: Optional predicted probability to display alongside the title.

    Returns:
        ImagePanel with a plain line chart as original_b64 and a
        contribution-colored scatter/line chart as explanation_b64.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(series)
    t = np.arange(T)
    vmax = float(np.abs(weights).max()) + 1e-8

    bg = "#f8f9fa"
    spine_color = "#dee2e6"

    def _save(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ---- Original: clean line chart ----------------------------------------
    fig_o, ax_o = plt.subplots(figsize=(5, 2.2))
    fig_o.patch.set_facecolor(bg)
    ax_o.set_facecolor(bg)
    ax_o.plot(t, series, color="#555", linewidth=1.2)
    ax_o.set_title("Original", fontsize=9, color="#2c3e50", pad=4)
    ax_o.set_xlabel("timestep", fontsize=7, color="#666")
    ax_o.tick_params(labelsize=7, colors="#666", length=3)
    for spine in ax_o.spines.values():
        spine.set_color(spine_color)
    ax_o.spines["top"].set_visible(False)
    ax_o.spines["right"].set_visible(False)
    ax_o.grid(True, alpha=0.3, linewidth=0.5, color=spine_color)
    fig_o.tight_layout(pad=0.5)
    orig_b64 = _save(fig_o)

    # ---- Explanation: line + per-timestep colored scatter -------------------
    fig_e, ax_e = plt.subplots(figsize=(5, 2.2))
    fig_e.patch.set_facecolor(bg)
    ax_e.set_facecolor(bg)
    ax_e.plot(t, series, color="#ccc", linewidth=0.8, zorder=1)

    for i in range(T):
        w = float(weights[i])
        alpha = min(abs(w) / vmax * 0.85 + 0.15, 1.0)
        color = "#27ae60" if w >= 0 else "#e74c3c"
        size = 10 + alpha * 30
        ax_e.scatter(i, series[i], c=color, alpha=alpha, s=size, zorder=2, linewidths=0)

    ax_e.set_title("Explanation", fontsize=9, color="#2c3e50", pad=4)
    ax_e.set_xlabel("timestep", fontsize=7, color="#666")
    ax_e.tick_params(labelsize=7, colors="#666", length=3)
    for spine in ax_e.spines.values():
        spine.set_color(spine_color)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.grid(True, alpha=0.3, linewidth=0.5, color=spine_color)
    fig_e.tight_layout(pad=0.5)
    exp_b64 = _save(fig_e)

    return ImagePanel(
        title=title,
        original_b64=orig_b64,
        explanation_b64=exp_b64,
        confidence=confidence,
    )
