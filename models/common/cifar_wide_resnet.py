import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float = 0.0,
        activate_before_residual: bool = False,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = float(drop_rate)
        self.equal_in_out = bool(in_planes == out_planes)
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        self.activate_before_residual = bool(activate_before_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_in_out and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return shortcut + out


class _WideNetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float,
        activate_before_residual: bool,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            *[
                _WideBasicBlock(
                    in_planes=in_planes if idx == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if idx == 0 else 1,
                    drop_rate=drop_rate,
                    activate_before_residual=activate_before_residual and idx == 0,
                )
                for idx in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class CifarWideResNetFeatures(nn.Module):
    """Shared CIFAR WideResNet trunk returning its final spatial feature map."""

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 8,
        drop_rate: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"Invalid WideResNet depth={depth}; expected (depth - 4) % 6 == 0.")
        if widen_factor <= 0:
            raise ValueError(f"WideResNet widen_factor must be > 0, got {widen_factor}.")
        if drop_rate < 0.0 or drop_rate >= 1.0:
            raise ValueError(f"WideResNet drop_rate must be in [0, 1), got {drop_rate}.")

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_layers = int((depth - 4) / 6)
        self.out_channels = int(channels[3])

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = _WideNetworkBlock(
            num_layers=num_layers,
            in_planes=channels[0],
            out_planes=channels[1],
            stride=1,
            drop_rate=drop_rate,
            activate_before_residual=True,
        )
        self.block2 = _WideNetworkBlock(
            num_layers=num_layers,
            in_planes=channels[1],
            out_planes=channels[2],
            stride=2,
            drop_rate=drop_rate,
            activate_before_residual=False,
        )
        self.block3 = _WideNetworkBlock(
            num_layers=num_layers,
            in_planes=channels[2],
            out_planes=channels[3],
            stride=2,
            drop_rate=drop_rate,
            activate_before_residual=False,
        )
        self.bn = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if initialize:
            self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / kernel_size))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.relu(self.bn(x))
