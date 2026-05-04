"""Utilities for rendering XAI image explanations as base64-encoded PNG panels."""
from __future__ import annotations

import base64
import io

import numpy as np

from .result import ImagePanel


def to_b64_png(img: np.ndarray) -> str:
    """Convert a float [0,1] or uint8 (H, W, 3) array to a base64-encoded PNG string."""
    from PIL import Image as PILImage

    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def lime_panel(
    image_rgb: np.ndarray,
    exp: object,
    label_idx: int,
    num_features: int,
    title: str,
    confidence: float | None = None,
) -> ImagePanel:
    """Build an ImagePanel by colorizing LIME superpixels green/red and adding boundaries.

    Args:
        image_rgb: (H, W, 3) float [0, 1] image.
        exp: LimeImageExplanation object (has .segments and .local_exp).
        label_idx: Class index whose local_exp to visualize.
        num_features: How many top superpixels to highlight.
        title: Panel heading shown in the HTML output.
        confidence: Optional predicted probability to display.

    Returns:
        ImagePanel with original and explanation images base64-encoded.
    """
    from skimage.segmentation import mark_boundaries

    segments: np.ndarray = exp.segments  # type: ignore[attr-defined]
    local_exp: list[tuple[int, float]] = exp.local_exp.get(label_idx, [])  # type: ignore[attr-defined]

    top = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features]

    # Dim everything to 35% brightness so non-highlighted regions fade out
    luma = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    overlay = np.stack([luma, luma, luma], axis=-1) * 0.35

    mask = np.zeros(segments.shape, dtype=int)
    for seg_id, weight in top:
        seg_mask = segments == seg_id
        mask[seg_mask] = 1
        original = image_rgb[seg_mask].astype(float)
        color = np.array([0.0, 1.0, 0.0]) if weight > 0 else np.array([1.0, 0.0, 0.0])
        overlay[seg_mask] = np.clip(original * 0.5 + color * 0.5, 0, 1)

    viz = mark_boundaries(np.clip(overlay, 0, 1), mask, color=(1.0, 1.0, 0.0))
    return ImagePanel(
        title=title,
        original_b64=to_b64_png(image_rgb),
        explanation_b64=to_b64_png(np.clip(viz, 0, 1)),
        confidence=confidence,
    )


def shap_panel(
    image: np.ndarray,
    shap_vals: np.ndarray,
    title: str,
    confidence: float | None = None,
) -> ImagePanel:
    """Build an ImagePanel by blending a RdYlGn heatmap of SHAP values onto the image.

    Args:
        image: (H, W, C) float [0, 1] image.
        shap_vals: (H, W, C) SHAP values for one output class (or regression).
        title: Panel heading shown in the HTML output.
        confidence: Optional predicted probability to display.

    Returns:
        ImagePanel with original and explanation images base64-encoded.
    """
    orig = image.astype(float)
    if orig.max() > 1.0:
        orig = orig / 255.0
    if orig.shape[-1] == 1:
        orig = np.repeat(orig, 3, axis=-1)

    heatmap = shap_vals.sum(axis=-1)                    # (H, W)
    vmax = np.abs(heatmap).max() + 1e-8

    # Alpha = contribution magnitude; near-zero stays transparent so background stays dark
    alpha = np.clip(np.abs(heatmap) / vmax, 0.0, 1.0)  # (H, W)

    # Green for positive contribution, red for negative
    color = np.where(
        heatmap[..., np.newaxis] >= 0,
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )  # (H, W, 3)

    alpha3 = alpha[..., np.newaxis]                     # (H, W, 1)
    blended = np.clip(orig * (1.0 - alpha3 * 0.85) + color * alpha3 * 0.85, 0, 1)
    return ImagePanel(
        title=title,
        original_b64=to_b64_png(orig),
        explanation_b64=to_b64_png(blended),
        confidence=confidence,
    )
