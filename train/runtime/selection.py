from typing import Mapping


BEST_SELECTION_MODES = ("topdown", "independent")


def metric_for_best(eval_metrics: Mapping[str, float], mode: str) -> float:
    """Select the checkpoint ranking score from validation metrics.

    Lexicographic order:
    1) FPA for the selected mode (higher is better)
    2) TICE for the selected mode (lower is better)
    3) wAP for the selected mode (higher is better)
    Falls back to deepest available level accuracy when H-CAST metrics are absent.
    """
    if mode not in BEST_SELECTION_MODES:
        raise ValueError(f"Unknown selection mode '{mode}'. Expected one of {BEST_SELECTION_MODES}.")

    fpa_key = f"fpa_{mode}"
    tice_key = f"tice_{mode}"
    wap_key = f"weighted_ap_{mode}"
    has_fpa = fpa_key in eval_metrics
    has_tice = tice_key in eval_metrics
    has_wap = wap_key in eval_metrics

    if has_fpa or has_tice or has_wap:
        fpa = float(eval_metrics.get(fpa_key, 0.0))
        # TICE is inconsistency rate, so lower is better: encode as -TICE.
        neg_tice = -float(eval_metrics.get(tice_key, 1.0))
        wap = float(eval_metrics.get(wap_key, 0.0))
        # Base-10 lexicographic packing for bounded metrics in [0, 1].
        return float(fpa + 1e-3 * neg_tice + 1e-6 * wap)

    prefix = f"acc_level_{mode}_"
    deepest = [
        key
        for key in eval_metrics
        if key.startswith(prefix) and key[len(prefix) :].isdigit()
    ]
    if not deepest:
        return float(eval_metrics.get(fpa_key, 0.0))
    deepest_key = max(deepest, key=lambda key: int(key.rsplit("_", 1)[-1]))
    primary = float(eval_metrics.get(deepest_key, 0.0))
    tie = float(eval_metrics.get(fpa_key, 0.0))
    # Tiny path-accuracy term stabilizes ordering when primary scores are tied.
    return float(primary + 1e-3 * tie)
