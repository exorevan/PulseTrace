"""Utilities for rendering XAI text explanations as base64-encoded PNG panels."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .result import ImagePanel

_BG = "#f8f9fa"
_CHAR_W = 0.010   # approximate axes-fraction width per character (monospace)
_SPACE_W = 0.005  # extra gap after each word
_LINE_H = 0.16    # vertical step between lines
_X0 = 0.02        # left margin
_Y0 = 0.90        # top start position
_X_MAX = 0.96     # right margin (wrap threshold)
_Y_MIN = 0.04     # bottom cutoff — words below this are clipped


def text_panel(
    text: str,
    word_weights: list[tuple[str, float]],
    title: str,
    confidence: float | None = None,
) -> ImagePanel:
    """Build an ImagePanel showing text with contribution-colored words.

    Args:
        text: Original text string to display.
        word_weights: (word, weight) pairs from LIME or SHAP. Positive weights
            render green, negative red; magnitude drives opacity. Words not in
            the list render neutral grey.
        title: Panel heading shown in the HTML output.
        confidence: Optional predicted probability displayed with the title.

    Returns:
        ImagePanel with a plain-text original and a word-colored explanation.
    """
    weight_map: dict[str, float] = {
        w.lower().strip(".,!?;:'\"") : v for w, v in word_weights
    }
    vmax = max((abs(v) for v in weight_map.values()), default=1e-8) + 1e-8
    words = text.split()

    def _save(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _render(ax, colorize: bool) -> None:
        x, y = _X0, _Y0
        for word in words:
            key = word.lower().strip(".,!?;:'\"")
            token = word + " "
            advance = len(token) * _CHAR_W + _SPACE_W
            if x + advance > _X_MAX and x > _X0:
                x = _X0
                y -= _LINE_H
            if y < _Y_MIN:
                break
            if colorize:
                w = weight_map.get(key)
                if w is None:
                    color, alpha = "#aaa", 0.5
                elif w >= 0:
                    alpha = min(abs(w) / vmax * 0.85 + 0.25, 1.0)
                    color = "#27ae60"
                else:
                    alpha = min(abs(w) / vmax * 0.85 + 0.25, 1.0)
                    color = "#e74c3c"
            else:
                color, alpha = "#555", 1.0
            ax.text(
                x, y, word,
                color=color, alpha=alpha,
                transform=ax.transAxes,
                fontsize=9, va="top", fontfamily="monospace",
            )
            x += advance

    fig_o, ax_o = plt.subplots(figsize=(7, 3))
    fig_o.patch.set_facecolor(_BG)
    ax_o.set_facecolor(_BG)
    ax_o.axis("off")
    ax_o.set_title("Original", fontsize=9, color="#2c3e50", pad=4)
    _render(ax_o, colorize=False)
    fig_o.tight_layout(pad=0.5)
    orig_b64 = _save(fig_o)

    fig_e, ax_e = plt.subplots(figsize=(7, 3))
    fig_e.patch.set_facecolor(_BG)
    ax_e.set_facecolor(_BG)
    ax_e.axis("off")
    ax_e.set_title("Explanation", fontsize=9, color="#2c3e50", pad=4)
    _render(ax_e, colorize=True)
    fig_e.tight_layout(pad=0.5)
    exp_b64 = _save(fig_e)

    return ImagePanel(
        title=title,
        original_b64=orig_b64,
        explanation_b64=exp_b64,
        confidence=confidence,
    )
