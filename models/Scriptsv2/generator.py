import torch
import torch.nn as nn
from utils import get_norm_layer
import functools


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ref_pad=1, use_act=True,
                 use_norm=True, norm_layer=nn.BatchNorm2d, down=True, **kwargs):
        super().__init__()

        layers = []
        if ref_pad > 0:
            layers.append(nn.ReflectionPad2d(ref_pad))

        layers.append(nn.Conv2d(in_channels, out_channels, **kwargs) if down
                      else nn.ConvTranspose2d(in_channels, out_channels, **kwargs))
        if use_norm:
            layers.append(norm_layer(out_channels))
        if use_act:
            layers.append(nn.ReLU(True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d, use_bias=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, norm_layer=norm_layer, bias=use_bias),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, norm_layer=norm_layer, bias=use_bias)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_residuals=9, norm="batch"):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        initial_layer = [ConvBlock(num_channels, num_features, 3, kernel_size=7, padding=0, bias=use_bias)]

        # add down-sampling
        down_sampling = []
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": use_bias}
        for i in range(2):  # add downsampling layers
            factor = 2 ** i
            down_sampling.append(ConvBlock(num_features * factor, num_features * factor * 2, 0, **kwargs))

        residuals = [ResidualBlock(num_features * 4, norm_layer, use_bias) for _ in range(num_residuals)]

        # add up-sampling
        up_sampling = []
        kwargs["output_padding"] = 1
        for i in range(2, 0, -1):  # add upsampling layers
            factor = 2 ** i
            up_sampling.append(ConvBlock(num_features * factor, num_features * factor // 2, 0, down=False, **kwargs))

        final_layer = [nn.ReflectionPad2d(3),
                       nn.Conv2d(num_features, num_channels, kernel_size=7, padding=0),
                       nn.Tanh()]

        layers = initial_layer + down_sampling + residuals + up_sampling + final_layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
