import torch
import torch.nn as nn
from utils import get_norm_layer
import functools
import math

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

        self.initial_layer = ConvBlock(num_channels, num_features, 3, kernel_size=7, padding=0, bias=use_bias)

        # add down-sampling
        down_sampling = []
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": use_bias}
        for i in range(2):
            factor = 2 ** i
            down_sampling.append(ConvBlock(num_features * factor, num_features * factor * 2, 0, **kwargs))
        self.down_sampling = nn.Sequential(*down_sampling)

        # residual transformers
        transformer_layer = nn.TransformerEncoderLayer(
            4096,
            8,
            2048,
            activation="gelu",
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_residuals)

        # add up-sampling
        up_sampling = []
        kwargs["output_padding"] = 1
        for i in range(2, 0, -1):
            factor = 2 ** i
            up_sampling.append(ConvBlock(num_features * factor, num_features * factor // 2, 0, down=False, **kwargs))
        self.up_sampling = nn.Sequential(*up_sampling)

        final_layer = [nn.ReflectionPad2d(3),
                       nn.Conv2d(num_features, num_channels, kernel_size=7, padding=0),
                       nn.Tanh()]
        self.final_layer = nn.Sequential(*final_layer)

    def reconstruct_image(self, windows, window_size=16, H=512, W=512):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.down_sampling(x)
        batch_size, channels, height, width = x.size()

        seq_len = height * width
        x = x.view(batch_size, channels, seq_len)  # reshape to (batch_size, channels, seq_len)
        pos_enc = self.get_positional_encoding(channels, seq_len, x.device)
        print(x.shape, pos_enc.shape)
        x = x + pos_enc
        print(x.shape)
        x = self.transformer_encoder(x)
        x = x - pos_enc
        x = x.view(batch_size, channels, height, width)  # reshape back to (batch_size, channels, height, width)

        x = self.up_sampling(x)
        x = self.final_layer(x)
        return x


    def get_positional_encoding(self, seq_len, embedding_dim, device):
        pos_enc = torch.zeros(seq_len, embedding_dim)

        # calculate angles for each position and dimension
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_dim))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        # add batch dimension
        pos_enc = pos_enc.unsqueeze(0)

        return pos_enc.to(device)