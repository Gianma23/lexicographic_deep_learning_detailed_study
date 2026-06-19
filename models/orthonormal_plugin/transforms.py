import torch.nn as nn


def get_activation(name: str, channels: int) -> nn.Module:
    if not isinstance(name, str):
        raise ValueError("Orthonormal plugin activation name must be a string.")
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "prelu":
        return nn.PReLU(num_parameters=int(channels))
    raise ValueError(f"Unsupported activation '{name}'.")


class PointResidualTransformationLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: str = "prelu",
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act1 = get_activation(activation, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = get_activation(activation, out_channels)
        if int(in_channels) == int(out_channels):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        residual = self.residual(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x + residual


class NarrowResidualTransformationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "prelu"):
        super().__init__()
        self.layer1 = PointResidualTransformationLayer(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            activation=activation,
        )
        self.layer2 = PointResidualTransformationLayer(
            in_channels=out_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            activation=activation,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def build_transformation_module(width: int, mode: str, owner: str = "Orthonormal plugin") -> nn.Module:
    if not isinstance(mode, str):
        raise ValueError(f"{owner} transform mode must be a string.")
    if mode == "full":
        return nn.Sequential(
            nn.BatchNorm1d(width),
            NarrowResidualTransformationHead(
                in_channels=width,
                out_channels=width,
                activation="prelu",
            ),
        )
    if mode == "bn_linear":
        return nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, width, bias=False),
            get_activation("prelu", width),
        )
    if mode == "final_only":
        return nn.Identity()
    raise ValueError(
        f"Unsupported {owner} transform_mode '{mode}'. "
        "Expected one of ['full', 'bn_linear', 'final_only']."
    )
