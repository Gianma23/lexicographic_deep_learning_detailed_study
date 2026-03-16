from typing import Optional, Sequence

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def supports_seeds() -> bool:
    return bool(
        cv2 is not None
        and hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "createSuperpixelSEEDS")
    )


def build_seeds_segments(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    num_superpixels: int = 196,
    num_levels: int = 1,
    prior: int = 2,
    histogram_bins: int = 5,
    double_step: bool = False,
    num_iterations: int = 15,
) -> Optional[torch.Tensor]:
    """Build SEEDS segment maps from a normalized `B x C x H x W` tensor."""
    if not supports_seeds():
        return None
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")

    bsz, channels, _, _ = images.shape
    if channels != 3:
        return None
    if len(mean) != channels or len(std) != channels:
        return None

    device = images.device
    x = images.detach().to(torch.float32)
    mean_t = torch.tensor(mean, dtype=x.dtype, device=device).view(1, channels, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=device).view(1, channels, 1, 1)
    x = (x * std_t + mean_t).clamp_(0.0, 1.0)

    x_u8 = (
        x.mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )

    segments = []
    for idx in range(bsz):
        rgb = x_u8[idx]
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width=int(lab.shape[1]),
            image_height=int(lab.shape[0]),
            image_channels=3,
            num_superpixels=int(num_superpixels),
            num_levels=int(num_levels),
            prior=int(prior),
            histogram_bins=int(histogram_bins),
            double_step=bool(double_step),
        )
        seeds.iterate(lab, num_iterations=int(num_iterations))
        seg = seeds.getLabels()
        segments.append(torch.from_numpy(seg.astype(np.int64, copy=False)))

    return torch.stack(segments, dim=0).to(device=device, dtype=torch.long)
