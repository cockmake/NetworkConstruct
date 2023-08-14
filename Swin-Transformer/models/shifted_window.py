import torch.nn as nn
import torch
from utils_func import windows_partition, windows_reverse
from models.window_attention import WindowAttention, MLP, PatchEmbedding, PatchMerging
def generate_mask(window_size, shift_size, input_resolution):
    H, W = input_resolution
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]

    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    cnt = 0


    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask = windows_partition(img_mask, window_size)

    mask = mask.reshape((-1, window_size * window_size))
    attn_mask = mask.unsqueeze(dim=-2) - mask.unsqueeze(dim=-1)
    attn_mask[attn_mask != 0] = float('-inf')
    return attn_mask

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 2 * dim)
        if shift_size > 0:
            self.attn_mask = generate_mask(window_size, shift_size, input_resolution)
            self.register_buffer('mask', self.attn_mask)
        else:
            self.mask = None


    def forward(self, x):
        # [B,  N, emb_dim]
        H, W = self.resolution
        B, N, Dim = x.shape

        x = x.reshape([B, H, W, Dim])
        h = x
        x = self.attn_norm(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=[1, 2])
        else:
            shifted_x = x

        x_windows = windows_partition(shifted_x, self.window_size)  # [B * N, ws, ws, dim]
        x_windows = x_windows.reshape((-1, self.window_size * self.window_size, Dim))
        attn_window = self.attn(x_windows, mask=self.mask)

        attn_window = attn_window.reshape((-1, self.window_size, self.window_size, Dim))
        shifted_x = windows_reverse(attn_window, H, W)

        # shift back
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=[1, 2])
        else:
            x = shifted_x
        x = h + x


        x = x.reshape((B, -1, Dim))
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x

if __name__ == '__main__':
    # 还可以添加relative_position_bias进一步增加性能
    data = torch.randn((8, 3, 224, 224))
    patch_emb = PatchEmbedding(4, 96)
    swin_block = SwinBlock(96, (224 // 4, 224 // 4), 8, 7, 7 // 2)
    patch_merge = PatchMerging((224 // 4, 224 // 4), 96)

    data = patch_emb(data)  # [B, N, dim]

    data = swin_block(data)

    data = patch_merge(data)
    print(data.shape)
