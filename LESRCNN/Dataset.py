import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import io
from PreProcess import PairedRandomCrop, PairedRandomHorizontalFlip, PairedRandomVerticalFlip, PairedRandomRotation, PairedNormalize
import os
from tqdm import tqdm


class SRDataset(Dataset):
    def __init__(self, scale=4, hr_patch_size=640, enhance_prob=0.5):
        super(SRDataset, self).__init__()
        self.lr_path_base = "../Datasets/DIV2K/DIV2K_train_LR"
        self.hr_path_base = "../Datasets/DIV2K/DIV2K_train_HR"
        self.lr_images_path = os.listdir(self.lr_path_base)
        self.hr_images_path = os.listdir(self.hr_path_base)
        self.preprocess = transforms.Compose([
            PairedRandomCrop(scale, hr_patch_size),
            PairedRandomHorizontalFlip(enhance_prob),
            PairedRandomVerticalFlip(enhance_prob),
            PairedRandomRotation(enhance_prob),
            PairedNormalize(mean=[0., 0., 0.], std=[255., 255., 255.], inplace=True)
        ])
    def __getitem__(self, idx):
        # assert self.lr_images_path[idx].startswith(self.lr_images_path[idx].split('.')[0]), "lr_images isn't corresponding to hr_images"
        lr_image = io.read_image(os.path.join(self.lr_path_base, self.lr_images_path[idx]),
                                 io.ImageReadMode.RGB)
        hr_image = io.read_image(os.path.join(self.hr_path_base, self.hr_images_path[idx]),
                                 io.ImageReadMode.RGB)
        # 进行预处理
        lr_image, hr_image = self.preprocess([lr_image, hr_image])
        return [lr_image, hr_image]
    def __len__(self):
        return len(self.hr_images_path)