import torch
import torch.nn as nn
from torchvision import datasets
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
class Generator(nn.Module):
    def __init__(self, input_feature=100):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # [B, input_feature, 1, 1] -> [B, 256, 4, 4]
            nn.ConvTranspose2d(input_feature, 256, (4, 4), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # [B, 256, 4, 4] -> [B, 128, 8, 8]
            nn.ConvTranspose2d(256, 128, (2, 2), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # [B, 128, 8, 8] -> [B, 64, 17, 17]
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # [B, 64, 17, 17] -> [B, 3, 32, 32]
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (2, 2)),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # [B, 3, 32, 32] -> [B, 64, 17, 17]
            nn.Conv2d(3, 64, (4, 4), (2, 2), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # [B, 64, 17, 17] -> [B, 128, 8, 8]
            nn.Conv2d(64, 128, (3, 3), (2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(128, 256, (2, 2), (2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # [B, 256, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(256, 1, (4, 4), (1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.dis(x)

epoches = 40
device = torch.device('cuda:0')
batch_size = 64
lr = 0.0001
preprocess = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.CIFAR10('.', train=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size, True, drop_last=True)
loss_fn = nn.MSELoss()
dis = Discriminator().to(device)
gen = Generator().to(device)
optimizerD = torch.optim.Adam(dis.parameters(), lr)
optimizerG = torch.optim.Adam(gen.parameters(), lr)

fixed_noise = torch.randn((10, 100, 1, 1), dtype=torch.float32, device=device)
# 可以称其为输入特征
real_label = torch.full((batch_size, 1, 1, 1), 1.0, device=device, dtype=torch.float32)
fake_label = torch.full((batch_size, 1, 1, 1), 0.0, device=device, dtype=torch.float32)
for epoch in range(epoches):
    tqdm_loader = tqdm(dataloader)
    for inp, _ in tqdm_loader:
        gen.train()
        inp = inp.to(device)
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optimizerD.zero_grad()
        real_output = dis(inp)
        real_loss = loss_fn(real_output, real_label)

        noise = torch.randn((batch_size, 100, 1, 1), dtype=torch.float32, device=device)
        fake_img = gen(noise)
        fake_output = dis(fake_img)
        fake_loss = loss_fn(fake_output, fake_label)

        error_loss = real_loss + fake_loss
        error_loss.backward()
        optimizerD.step()
        optimizerD.zero_grad()

        # Update G network: maximize log(D(G(z)))
        optimizerG.zero_grad()
        noise = torch.randn((batch_size, 100, 1, 1), dtype=torch.float32, device=device)
        fake = gen(noise)
        output = dis(fake)
        loss = loss_fn(output, real_label)
        loss.backward()
        optimizerG.step()
        optimizerG.zero_grad()

        tqdm_loader.set_description(f"dis_loss: {format(fake_loss.cpu().item(), '.4f')} "
                                    f"gen_loss: {format(loss.cpu().item(), '.4f')}")
    # visualize
    gen.eval()
    with torch.no_grad():
        generated_image = gen(fixed_noise).cpu().numpy()
        plt.figure(figsize=(10, 5))
        for i in range(10):
            image = generated_image[i]
            image = np.clip(image, 0.0, 1.0)
            image = image.transpose((1, 2, 0))
            plt.subplot(1, 10, i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()