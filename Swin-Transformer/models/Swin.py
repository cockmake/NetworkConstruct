import torch.nn as nn

from models.shifted_window import SwinBlock, PatchMerging, PatchEmbedding


class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, patch_merging=False):
        super(SwinStage, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim, input_resolution, num_heads, window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2
                )
            )
        if patch_merging:
            self.blocks.append(PatchMerging(input_resolution, dim))

    def forward(self, x):
        x = self.blocks(x)
        return x


class Swin(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 window_size,
                 num_heads=(3, 6, 12, 24),
                 depths=(2, 2, 6, 2),
                 num_classes=1000):
        super(Swin, self).__init__()
        if num_heads is None:
            num_heads = [8, 8, 8, 8]
        self.num_classes = num_classes
        self.depths = depths
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.patch_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_features = embed_dim * 2 ** (self.num_stages - 1)
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.stages = nn.Sequential()
        for idx, (depth, num_head) in enumerate(zip(depths, num_heads)):
            self.stages.append(
                SwinStage(
                    int(self.embed_dim * 2 ** idx),
                    (self.patch_resolution[0] // (2 ** idx), self.patch_resolution[1] // (2 ** idx)),
                    depth,
                    num_head,
                    window_size,
                    True if idx < self.num_stages - 1 else False
                )
            )

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.stages(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        # [B, dim, num_windows]
        x = self.avgpool(x)
        # [B, dim, 1]
        x = x.flatten(start_dim=1)
        return self.fc(x)
