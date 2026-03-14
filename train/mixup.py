from typing import Any, Dict, Optional, Tuple

import torch


def mixup_alpha(cfg: Any) -> float:
    # H-CAST uses `mixup`; keep `mixup_alpha` as backward-compatible alias.
    train_cfg = cfg.train
    raw = train_cfg.get("mixup", train_cfg.get("mixup_alpha", 0.0))
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        alpha = 0.0
    return max(0.0, alpha)


def mixup_prob(cfg: Any) -> float:
    raw = cfg.train.get("mixup_prob", 1.0)
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        prob = 1.0
    return min(max(prob, 0.0), 1.0)


def cutmix_alpha(cfg: Any) -> float:
    raw = cfg.train.get("cutmix", 0.0)
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        alpha = 0.0
    return max(0.0, alpha)


def mixup_switch_prob(cfg: Any) -> float:
    raw = cfg.train.get("mixup_switch_prob", 0.5)
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        prob = 0.5
    return min(max(prob, 0.0), 1.0)


def mixup_mode(cfg: Any) -> str:
    mode = str(cfg.train.get("mixup_mode", "batch")).strip().lower()
    return mode or "batch"


def cutmix_minmax(cfg: Any) -> Optional[Tuple[float, float]]:
    raw = cfg.train.get("cutmix_minmax", None)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        lo = float(raw[0])
        hi = float(raw[1])
    except (TypeError, ValueError):
        return None
    lo = min(max(lo, 0.0), 1.0)
    hi = min(max(hi, 0.0), 1.0)
    if hi <= lo:
        return None
    return lo, hi


def blend_metrics(a: Dict[str, float], b: Dict[str, float], lam: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in set(a.keys()) | set(b.keys()):
        va = float(a.get(key, 0.0))
        vb = float(b.get(key, 0.0))
        out[key] = float(lam * va + (1.0 - lam) * vb)
    return out


def _sample_beta(alpha: float, device: torch.device) -> float:
    return float(torch.distributions.Beta(alpha, alpha).sample(()).to(device=device).item())


def _rand_bbox(img_h: int, img_w: int, lam: float, device: torch.device) -> Tuple[int, int, int, int]:
    ratio = (1.0 - lam) ** 0.5
    cut_h = int(img_h * ratio)
    cut_w = int(img_w * ratio)
    cy = int(torch.randint(0, img_h, (), device=device).item())
    cx = int(torch.randint(0, img_w, (), device=device).item())
    yl = max(0, cy - cut_h // 2)
    yh = min(img_h, cy + cut_h // 2)
    xl = max(0, cx - cut_w // 2)
    xh = min(img_w, cx + cut_w // 2)
    return yl, yh, xl, xh


def _rand_bbox_minmax(
    img_h: int, img_w: int, ratio_minmax: Tuple[float, float], device: torch.device
) -> Tuple[int, int, int, int]:
    lo, hi = ratio_minmax
    cut_h = int(torch.randint(int(img_h * lo), max(int(img_h * hi), int(img_h * lo) + 1), (), device=device).item())
    cut_w = int(torch.randint(int(img_w * lo), max(int(img_w * hi), int(img_w * lo) + 1), (), device=device).item())
    yl = int(torch.randint(0, max(1, img_h - cut_h + 1), (), device=device).item())
    xl = int(torch.randint(0, max(1, img_w - cut_w + 1), (), device=device).item())
    yh = min(img_h, yl + cut_h)
    xh = min(img_w, xl + cut_w)
    return yl, yh, xl, xh


def apply_mixup(
    images: torch.Tensor, labels: torch.Tensor, cfg: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    mix_alpha = mixup_alpha(cfg)
    cut_alpha = cutmix_alpha(cfg)
    cut_minmax = cutmix_minmax(cfg)
    prob = mixup_prob(cfg)
    switch_prob = mixup_switch_prob(cfg)
    mode = mixup_mode(cfg)
    cutmix_active = cut_alpha > 0.0 or cut_minmax is not None
    mixup_active = mix_alpha > 0.0
    if (not mixup_active and not cutmix_active) or prob <= 0.0 or mode != "batch":
        return images, labels, labels, 1.0, False

    if torch.rand((), device=images.device).item() >= prob:
        return images, labels, labels, 1.0, False

    use_cutmix = False
    if mixup_active and cutmix_active:
        use_cutmix = bool(torch.rand((), device=images.device).item() < switch_prob)
    elif cutmix_active:
        use_cutmix = True

    alpha_for_sampling = cut_alpha if use_cutmix else mix_alpha
    if cut_minmax is not None and use_cutmix:
        alpha_for_sampling = 1.0
    if alpha_for_sampling <= 0.0:
        alpha_for_sampling = 1.0

    lam = _sample_beta(alpha_for_sampling, images.device)

    # H-CAST pairs samples with reversed batch order in batch mode.
    images_b = images.flip(0)
    labels_b = labels.flip(0)

    if use_cutmix:
        _, _, img_h, img_w = images.shape
        if cut_minmax is not None:
            yl, yh, xl, xh = _rand_bbox_minmax(img_h, img_w, cut_minmax, images.device)
        else:
            yl, yh, xl, xh = _rand_bbox(img_h, img_w, lam, images.device)
        mixed = images.clone()
        mixed[:, :, yl:yh, xl:xh] = images_b[:, :, yl:yh, xl:xh]
        area = max(0, yh - yl) * max(0, xh - xl)
        lam = float(1.0 - float(area) / float(img_h * img_w))
    else:
        mixed = images * lam + images_b * (1.0 - lam)

    return mixed, labels, labels_b, lam, True
