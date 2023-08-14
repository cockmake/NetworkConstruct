import torch

from models.Swin import Swin

img_size = 256
patch_size = 4
embed_dim = 96
window_size = 8
# img_size // (patch_size * window_size)
data = torch.randn((16, 3, img_size, img_size))
model = Swin(img_size, patch_size, embed_dim, window_size)
print(model(data).shape)
