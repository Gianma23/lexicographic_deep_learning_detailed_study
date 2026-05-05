from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _model_name(cfg: Any) -> str:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None or not hasattr(model_cfg, "get"):
        return ""
    return str(model_cfg.get("name", "")).strip().lower()


def _transforms_cfg(cfg: Any) -> Any:
    dataset_cfg = getattr(cfg, "dataset", {}) or {}
    transforms_cfg = dataset_cfg.get("transforms", {}) if hasattr(dataset_cfg, "get") else {}
    return transforms_cfg or {}


def mixup_alpha(cfg: Any) -> float:
    raw = _transforms_cfg(cfg).get("mixup", 0.0)
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        alpha = 0.0
    return max(0.0, alpha)


def cutmix_alpha(cfg: Any) -> float:
    raw = _transforms_cfg(cfg).get("cutmix", 0.0)
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        alpha = 0.0
    return max(0.0, alpha)


def cutmix_minmax(cfg: Any) -> Optional[Tuple[float, float]]:
    raw = _transforms_cfg(cfg).get("cutmix_minmax", None)
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 2:
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


def mixup_prob(cfg: Any) -> float:
    raw = _transforms_cfg(cfg).get("mixup_prob", 1.0)
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        prob = 1.0
    return min(max(prob, 0.0), 1.0)


def mixup_switch_prob(cfg: Any) -> float:
    raw = _transforms_cfg(cfg).get("mixup_switch_prob", 0.5)
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        prob = 0.5
    return min(max(prob, 0.0), 1.0)


def mixup_mode(cfg: Any) -> str:
    mode = str(_transforms_cfg(cfg).get("mixup_mode", "batch")).strip().lower()
    return mode or "batch"


def one_hot(x: torch.Tensor, num_classes: int, on_value: float = 1.0, off_value: float = 0.0) -> torch.Tensor:
    x = x.long().view(-1, 1)
    return torch.full((x.size(0), num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def mixup_target(target: torch.Tensor, num_classes: int, lam=1.0, smoothing: float = 0.0) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape: Sequence[int], lam: float, margin: float = 0.0, count=None):
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape: Sequence[int], minmax: Tuple[float, float], count=None):
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(
    img_shape: Sequence[int],
    lam: float,
    ratio_minmax: Optional[Tuple[float, float]] = None,
    correct_lam: bool = True,
    count=None,
):
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """Mixup/Cutmix copied from upstream H-CAST mixup_hier.py."""

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 0.0,
        cutmix_minmax: Optional[Tuple[float, float]] = None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        correct_lam: bool = True,
        label_smoothing: float = 0.1,
        num_classes: Optional[List[int]] = None,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes or []
        self.mode = mode
        self.correct_lam = correct_lam

    def _params_per_elem(self, batch_size: int):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
            use_cutmix = np.random.rand(batch_size) < self.switch_prob
            lam_mix = np.where(
                use_cutmix,
                np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
            )
        elif self.mixup_alpha > 0.0:
            lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        elif self.cutmix_alpha > 0.0:
            use_cutmix = np.ones(batch_size, dtype=bool)
            lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
        else:
            raise AssertionError(
                "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            )
        lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                raise AssertionError(
                    "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
                )
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        float(lam),
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1.0 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape,
                        float(lam),
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam,
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1.0 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1.0 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x: torch.Tensor):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam,
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x: torch.Tensor, target: List[torch.Tensor]):
        assert len(x) % 2 == 0, "Batch size should be even when using this"
        if self.mode == "elem":
            lam = self._mix_elem(x)
        elif self.mode == "pair":
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)

        mixed_targets = []
        for level, level_target in enumerate(target):
            mixed_targets.append(
                mixup_target(level_target, int(self.num_classes[level]), lam, self.label_smoothing)
            )
        return (x, *mixed_targets)


def build_mixup_fn(cfg: Any, num_classes_per_level: Optional[List[int]] = None) -> Optional[Mixup]:
    mixup_active = (
        mixup_alpha(cfg) > 0.0
        or cutmix_alpha(cfg) > 0.0
        or cutmix_minmax(cfg) is not None
    )
    if mixup_active and _model_name(cfg) == "hiercos":
        raise ValueError("Hier-COS does not support mixup/cutmix. Set dataset.transforms.mixup/cutmix to 0.")
    if not mixup_active:
        return None
    if not num_classes_per_level:
        raise ValueError("Mixup/CutMix requires `num_classes_per_level`.")
    return Mixup(
        mixup_alpha=mixup_alpha(cfg),
        cutmix_alpha=cutmix_alpha(cfg),
        cutmix_minmax=cutmix_minmax(cfg),
        prob=mixup_prob(cfg),
        switch_prob=mixup_switch_prob(cfg),
        mode=mixup_mode(cfg),
        label_smoothing=float(cfg.train.get("smoothing", 0.1)),
        num_classes=[int(x) for x in num_classes_per_level],
    )
