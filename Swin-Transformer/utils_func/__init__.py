import torch
from torch import Tensor
def windows_partition(x: Tensor, window_size):
    B, H, W, Dim = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, Dim))
    x = torch.permute(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape((-1, window_size, window_size, Dim))
    return x

def windows_reverse(windows, H, W):
    _, ws, ws, dim = windows.shape
    B = int(windows.shape[0] // (H / ws * W / ws))
    x = windows.reshape((B, H // ws, W // ws, ws, ws, dim))
    x = x.transpose(2, 3)  # [B, H // ws, ws, W // ws, ws, dim]
    x = x.reshape((B, H, W, dim))
    return x  # [B, H, W, dim]