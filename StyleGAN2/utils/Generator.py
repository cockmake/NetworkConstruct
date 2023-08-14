# <!todo> i am try to understand the construct of StyleGAN2
import torch.nn as nn
import torch
from torch.nn import functional as F
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.randn((channel,)))
        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)
def fused_leaky_relu(x, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        # <!to understand why rest_dim ?>
        rest_dim = [1] * (len(x.shape) - len(bias.shape) - 1)
        return F.leaky_relu(x + bias.reshape((1, bias.shape[0], *rest_dim)), negative_slope=negative_slope) * scale
    else:
        return F.leaky_relu(x, negative_slope=negative_slope) * scale
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    @staticmethod
    def forward(x):
        # 这里的 h * w 代表的是某一个像素的特征？
        # x: [B, channels, h, w]
        # rsqrt 平方根然后取倒数
        return x * torch.rsqrt(
            torch.mean(x * x, dim=1, keepdim=True) + torch.tensor([1e-8], dtype=x.dtype, device=x.device)
        )
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias))
        else:
            self.bias = None

        self.lr_mul = lr_mul
        self.scale = lr_mul / (in_dim ** 0.5)

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x):
        batch = x.shape[0]
        return self.input.tile((batch, 1, 1, 1))
class NoiseInjection(nn.Module):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.tensor([1], dtype=torch.float32))
    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = torch.randn((batch, 1, height, width), device=image.device)
        return image + self.weight * noise