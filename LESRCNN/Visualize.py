import os
import torch
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from torchvision import io
from torchvision.transforms import transforms as T
def showValImage(tb_writer: tensorboard.SummaryWriter, network, val_image_path, step, device):
    fig = getValFig(network, val_image_path, device=device)
    tb_writer.add_figure('val_image', fig, step)

def getValFig(network, val_image_path, device):
    network.eval()
    val_images = os.listdir(val_image_path)
    num_image = len(val_images)
    with torch.no_grad():
        t = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.])
        ])
        fig = plt.figure(figsize=(1, num_image), dpi=2048)
        for i, img in enumerate(val_images):
            image = t(io.read_image(os.path.join(val_image_path, img), io.ImageReadMode.RGB).to(device).float().unsqueeze(dim=0))
            output_image = torch.round(torch.clip(network(image).permute(0, 2, 3, 1) * 255.0, min=0, max=255))\
                .cpu().numpy().astype('uint8')  # plt支持显示0-1
            plt.subplot(num_image, 1, i + 1, xticks=[], yticks=[])
            plt.imshow(output_image[0])
        fig.show()
    return fig