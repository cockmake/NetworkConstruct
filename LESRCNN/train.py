import os

import torch
import random

from tqdm import tqdm

from Generator import LESRCNNGenerator
from torch.utils.data import DataLoader
from Dataset import SRDataset
from torch.utils import tensorboard
from Visualize import showValImage

# seed = 7230
# torch.random.manual_seed(seed)
# random.seed(seed)

hr_patch_size = 640
num_workers = 4
batch_size = 4
scale = 4
use_cuda = True
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
inp_channels = 3
inp_height = hr_patch_size // scale
inp_width = hr_patch_size // scale
lr = 0.0001
epoches = 160  # 这里主要与lr_scheduler相对应

if __name__ == '__main__':
    # writer写在主进程里
    num = len(os.listdir('checkpoints'))
    writer = tensorboard.SummaryWriter('checkpoints/lesrcnn' + ('' if num == 0 else str(num + 1)))
    # 初始化模型
    gen = LESRCNNGenerator()
    gen.load_state_dict(torch.load('trained_lesrcnn.pt'))
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(gen.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)

    # 写入模型Graph
    init_data = torch.randn((1, 3, hr_patch_size // scale, hr_patch_size // scale))
    writer.add_graph(gen, init_data)
    # 加载数据集
    train_dataset = SRDataset(hr_patch_size=hr_patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    # 训练步长记录
    global_record_step = 0
    epoch_train_step = 0
    record_interval = 1
    epoch_loss = 0
    # 正式训练部分
    gen.to(device)
    for epoch in range(epoches):
        loader = tqdm(train_dataloader)
        epoch_loss = 0
        epoch_train_step = 0
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            epoch_train_step += 1
            optimizer.zero_grad()
            output = gen(lr_img)
            loss = loss_fn(output, hr_img)
            loss.backward()
            optimizer.step()
            record_loss = loss.cpu().item()
            epoch_loss += record_loss
            loader.set_description(f"loss: {format(record_loss, '.4f')}")
            if epoch_train_step % record_interval == 0:
                writer.add_scalar('train_step_loss', record_loss, global_record_step)
                global_record_step += 1
        epoch_loss /= epoch_train_step
        writer.add_scalar('epoch_loss', epoch_loss, epoch)
        showValImage(writer, gen, 'val_images', epoch, device=device)
        lr_scheduler.step()
        torch.save(gen.state_dict(), f"trained/lesrcnn_{epoch}.pt")
    writer.close()

