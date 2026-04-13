from __future__ import annotations

from itertools import groupby

from pulsetrace.explainers.result import ExplanationResult, FeatureContribution

_BAR_WIDTH = 20
_SCALE = 0.5  # weight at which bar is 100% filled


def render(result: ExplanationResult) -> None:
    """Print an ExplanationResult as a formatted table to stdout."""
    print(f"\n{'=' * 60}")
    print(
        f"  {result.method.upper()} | mode={result.mode} | task={result.task}"
    )
    print(f"  Target: {result.target_name}")
    print(f"{'=' * 60}")

    if result.task == "regression":
        _print_contributions(result.contributions)
    else:
        sorted_c = sorted(result.contributions, key=lambda c: str(c.label))
        for label, group in groupby(sorted_c, key=lambda c: c.label):
            print(f"\n  Class: {label}")
            _print_contributions(list(group), indent=4)

    print()


def _print_contributions(
    contributions: list[FeatureContribution], indent: int = 2
) -> None:
    pad = " " * indent
    by_abs = sorted(contributions, key=lambda c: abs(c.weight), reverse=True)
    for fc in by_abs:
        bar = _make_bar(fc.weight)
        print(f"{pad}{fc.feature:<32} {fc.weight:+.4f}  {bar}")


def _make_bar(weight: float) -> str:
    filled = int(min(abs(weight) / _SCALE, 1.0) * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    char = "#" if weight >= 0 else "-"
    return "[" + char * filled + "." * empty + "]"
