import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.cov = nn.Conv2d(3, emb_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.cov(x)  # [B, emb_dim, H', W']
        x = torch.permute(x, (0, 2, 3, 1))  # [B, H', W', emb_dim]
        x = self.norm(x)

        x = x.flatten(start_dim=1, end_dim=2)  # [B, N, emb_dim]
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, emb_dim):
        super(PatchMerging, self).__init__()
        self.resolution = input_resolution
        self.dim = emb_dim
        self.reduction = nn.Linear(4 * emb_dim, 2 * emb_dim)
        self.norm = nn.LayerNorm(4 * emb_dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, emb_dim = x.shape

        x = x.reshape([b, h, w, emb_dim])

        x0 = x[:, 0::2, 0::2, :].flatten(start_dim=1, end_dim=2)
        x1 = x[:, 0::2, 1::2, :].flatten(start_dim=1, end_dim=2)
        x2 = x[:, 1::2, 0::2, :].flatten(start_dim=1, end_dim=2)
        x3 = x[:, 1::2, 1::2, :].flatten(start_dim=1, end_dim=2)
        #  [N, H' / merge_size, W' / merge_size, emb_dim]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [N, H' / merge_size, W' / merge_size, 4 * emb_dim]
        x = self.reduction(self.norm(x))
        x = x.reshape((b, -1, 2 * emb_dim))
        return x


class MLP(nn.Module):
    def __init__(self, emb_dim, ffc, dropout=0.1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(emb_dim, ffc)
        self.fc2 = nn.Linear(ffc, emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 先标准化再激活
        x = self.fc2(self.gelu(self.fc(x)))
        x = self.dropout(self.gelu(x))
        return x


class WindowAttention(nn.Module):
    def __init__(self, emb_dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def transpose_multi_head(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)  # [N, num_patches, num_heads, head_dim]
        x = x.transpose(1, 2)  # [N, num_heads, num_patches, head_dim]
        return x

    def forward(self, x, mask=None):
        # [N, num_patches, emb_dim]
        B, N, dim = x.shape
        q, k, v = torch.split(self.qkv(x), [self.emb_dim] * 3, dim=-1)
        q, k, v = list(map(self.transpose_multi_head, [q, k, v]))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [n_windows, n_patches, n_patches]
            # attn: [B * n_windows, num_heads, num_patches, num_patches]
            attn = attn.reshape((B // mask.shape[0]), mask.shape[0], self.num_heads, N, N)
            # attn: [B, n_windows, num_heads, num_patches, num_patches]
            attn += mask.unsqueeze(dim=1)
            attn = attn.reshape(-1, self.num_heads, N, N)
            # attn: [B * n_windows, num_heads, num_patches, num_patches]
        out = torch.matmul(attn, v)  # [N, num_heads, num_patches, head_dim]
        out = out.transpose(1, 2).flatten(start_dim=2)
        return self.proj(out)
print(torch.__version__)