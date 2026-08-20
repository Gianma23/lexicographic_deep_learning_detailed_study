import math
from typing import Any, Dict, Mapping, Sequence, Tuple


BEST_SELECTION_MODES = ("topdown", "independent")
BEST_SELECTION_STRATEGIES = ("hierarchical_metrics", "deepest_accuracy")
SelectionKey = Tuple[float, float, float]


def _finite_or_worst(value: Any, *, negate: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if not math.isfinite(parsed):
        return float("-inf")
    return float(-parsed if negate else parsed)


def _deepest_accuracy_key(eval_metrics: Mapping[str, float], mode: str) -> str:
    prefix = f"acc_level_{mode}_"
    candidates = [
        key
        for key in eval_metrics
        if key.startswith(prefix) and key[len(prefix) :].isdigit()
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda key: int(key.rsplit("_", 1)[-1]))


def selection_key(
    eval_metrics: Mapping[str, float],
    mode: str,
    strategy: str = "hierarchical_metrics",
) -> SelectionKey:
    """Return an exact tuple used for validation-checkpoint comparison.

    ``hierarchical_metrics`` uses the lexicographic order FPA (higher), TICE
    (lower), then weighted AP (higher). ``deepest_accuracy`` uses only the
    deepest-level accuracy for the selected decoder, matching single-output
    validation monitors such as the released HT-CapsNet checkpoint callback.

    Missing or non-finite values rank below every finite value. Under
    ``hierarchical_metrics``, deepest-level accuracy is the fallback when none
    of the three hierarchy metrics is present.
    """
    if mode not in BEST_SELECTION_MODES:
        raise ValueError(f"Unknown selection mode '{mode}'. Expected one of {BEST_SELECTION_MODES}.")
    if strategy not in BEST_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unknown checkpoint-selection strategy '{strategy}'. "
            f"Expected one of {BEST_SELECTION_STRATEGIES}."
        )

    deepest_key = _deepest_accuracy_key(eval_metrics, mode)
    if strategy == "deepest_accuracy":
        primary = (
            _finite_or_worst(eval_metrics.get(deepest_key))
            if deepest_key
            else float("-inf")
        )
        return primary, float("-inf"), float("-inf")

    fpa_key = f"fpa_{mode}"
    tice_key = f"tice_{mode}"
    wap_key = f"weighted_ap_{mode}"
    if any(key in eval_metrics for key in (fpa_key, tice_key, wap_key)):
        return (
            _finite_or_worst(eval_metrics.get(fpa_key)),
            _finite_or_worst(eval_metrics.get(tice_key), negate=True),
            _finite_or_worst(eval_metrics.get(wap_key)),
        )

    primary = _finite_or_worst(eval_metrics.get(deepest_key)) if deepest_key else float("-inf")
    return primary, float("-inf"), float("-inf")


def is_better(candidate: Sequence[float], incumbent: Sequence[float]) -> bool:
    """Compare two normalized three-component keys using Python tuple ordering."""
    return normalize_selection_key(candidate) > normalize_selection_key(incumbent)


def normalize_selection_key(value: Any, *, legacy_primary: Any = None) -> SelectionKey:
    """Normalize serialized keys and upgrade legacy scalar best metrics."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(_finite_or_worst(component) for component in value)  # type: ignore[return-value]
    if legacy_primary is not None:
        return _finite_or_worst(legacy_primary), float("-inf"), float("-inf")
    if isinstance(value, (int, float)):
        return _finite_or_worst(value), float("-inf"), float("-inf")
    return float("-inf"), float("-inf"), float("-inf")


def selection_components(
    eval_metrics: Mapping[str, float],
    mode: str,
    strategy: str = "hierarchical_metrics",
) -> Dict[str, Any]:
    """Return a JSON/YAML-friendly description of a selection decision."""
    key = selection_key(eval_metrics, mode, strategy=strategy)
    fpa_key = f"fpa_{mode}"
    if strategy == "deepest_accuracy":
        primary_name = _deepest_accuracy_key(eval_metrics, mode) or fpa_key
    elif any(
        metric_key in eval_metrics
        for metric_key in (fpa_key, f"tice_{mode}", f"weighted_ap_{mode}")
    ):
        primary_name = fpa_key
    else:
        primary_name = _deepest_accuracy_key(eval_metrics, mode) or fpa_key
    return {
        "primary_name": primary_name,
        "primary": key[0],
        "neg_tice": key[1],
        "weighted_ap": key[2],
        "key": list(key),
    }


def metric_for_best(
    eval_metrics: Mapping[str, float],
    mode: str,
    strategy: str = "hierarchical_metrics",
) -> float:
    """Compatibility wrapper returning only the primary scheduler metric."""
    return float(selection_key(eval_metrics, mode, strategy=strategy)[0])
