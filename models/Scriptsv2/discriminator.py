import torch
import torch.nn as nn
import functools
from utils import get_norm_layer


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 kernel=4, stride=1, padding=1, use_norm=True, **kwargs):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, **kwargs),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        ]

        if not use_norm:
            layers.pop(1)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), norm="batch"):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # initial block doesn't use InstanceNorm
        layers = [Block(in_channels, features[0], stride=2, use_norm=False)]
        for (idx, feature) in enumerate(features[1:]):
            stride = 2 if feature != features[-1] else 1
            layers.append(Block(features[idx], feature, stride=stride, bias=use_bias))

        layers.append(nn.Conv2d(features[-1],
                                1,
                                kernel_size=4,
                                stride=1,
                                padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # maybe use sigmoid?

