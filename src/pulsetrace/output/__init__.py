from __future__ import annotations

from pulsetrace.explainers.result import ExplanationResult

from .console import render as _render_console
from .html_renderer import render as _render_html
from .json_renderer import render as _render_json


def render(result: ExplanationResult, output_format: str = "console") -> None:
    if output_format == "json":
        _render_json(result)
    elif output_format == "html":
        _render_html(result)
    else:
        _render_console(result)


__all__ = ["render"]
