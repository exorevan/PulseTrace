from __future__ import annotations

import re
from itertools import groupby

from pulsetrace.explainers.result import ExplanationResult, FeatureContribution

_BAR_WIDTH = 20
_SCALE = 0.5  # weight at which bar is 100% filled


def render(result: ExplanationResult) -> None:
    """Print an ExplanationResult as a formatted table to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {result.method.upper()} | mode={result.mode} | task={result.task}")
    print(f"  Target: {result.target_name}")
    if result.global_samples is not None:
        print(f"  Samples: {result.global_samples}")
    print(f"{'=' * 60}")

    if result.task == "regression":
        base = result.base_values.get(None) if result.base_values else None
        if base is not None:
            print(f"\n  Base value: {base:+.4f}")
        _print_contributions(result.contributions)
    else:
        sorted_c = sorted(result.contributions, key=lambda c: str(c.label))
        groups = [(label, list(group)) for label, group in groupby(sorted_c, key=lambda c: c.label)]
        for i, (label, group) in enumerate(groups):
            if i > 0:
                print(f"\n  {'- ' * 18}")
            base = result.base_values.get(label) if result.base_values else None
            base_str = f"  (base: {base:+.4f})" if base is not None else ""
            print(f"\n  Class: {label}{base_str}")
            _print_contributions(group, indent=4)

    print()


def _importance_label(weight: float) -> str:
    if abs(weight) < _SCALE * 0.15:
        return "(not important)"
    if weight < 0:
        return "(lowers result)"
    return "(raises result)"


def _param_name(feature: str) -> str:
    """Extract the base parameter name from a LIME condition string.

    E.g. '1.60 < PetalLengthCm <= 4.35' → 'PetalLengthCm'
         'PetalWidthCm <= 0.30'          → 'PetalWidthCm'
    """
    m = re.search(r"[A-Za-z_]\w*", feature)
    return m.group(0) if m else feature


def _condition_lower_bound(feature: str) -> float:
    """Return the lower bound of a LIME condition for numeric sorting within a parameter group.

    'value < feature ...'  → value   (range starts at value)
    'feature > value'      → value   (open-ended upper range)
    'feature <= value'     → -inf    (lowest range, no lower bound)
    """
    m = re.match(r"^([\d.]+)\s*<", feature)
    if m:
        return float(m.group(1))
    m = re.search(r">\s*([\d.]+)", feature)
    if m:
        return float(m.group(1))
    return float("-inf")


def _print_contributions(
    contributions: list[FeatureContribution], indent: int = 2
) -> None:
    pad = " " * indent
    by_name = sorted(contributions, key=lambda c: (_param_name(c.feature), _condition_lower_bound(c.feature)))
    prev_param = None
    for fc in by_name:
        current_param = _param_name(fc.feature)
        if prev_param is not None and current_param != prev_param:
            print()
        prev_param = current_param
        bar = _make_bar(fc.weight)
        label = _importance_label(fc.weight)
        print(f"{pad}{fc.feature:<32} {fc.weight:+.4f}  {bar}  {label}")


def _make_bar(weight: float) -> str:
    filled = int(min(abs(weight) / _SCALE, 1.0) * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    char = "#" if weight >= 0 else "-"
    return "[" + char * filled + "." * empty + "]"
