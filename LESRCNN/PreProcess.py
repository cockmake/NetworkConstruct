import random
import torch
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as TF
class PairedRandomCrop:
    def __init__(self, scale, hr_patch_size=640):
        """
        :param scale: model upscale factor.
        :param hr_patch_size: cropped gt patch size.
        """
        self.hr_patch_size = hr_patch_size
        self.scale = scale
    def __call__(self, lr_hr):
        assert len(lr_hr) == 2, "please input paired data"
        lr_patch_size = self.hr_patch_size // self.scale
        _, h_lr, w_lr = lr_hr[0].shape
        _, h_hr, w_hr = lr_hr[1].shape
        if h_lr * self.scale != h_hr or w_lr * self.scale != w_hr:
            raise ValueError('scale size not match')
        if h_lr < lr_patch_size or w_lr < lr_patch_size:
            raise ValueError('too big hr_patch_size')
        top = random.randint(0, h_lr - lr_patch_size)
        left = random.randint(0, w_lr - lr_patch_size)
        lr_image = lr_hr[0][:, top: top + lr_patch_size, left: left + lr_patch_size]
        top_hr, left_hr = top * self.scale, left * self.scale
        hr_image = lr_hr[1][:, top_hr: top_hr + self.hr_patch_size, left_hr: left_hr + self.hr_patch_size]
        return [lr_image, hr_image]
class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, prob=0.5):
        super(PairedRandomHorizontalFlip, self).__init__(prob)
    def forward(self, image):
        if random.random() < self.p:
            if isinstance(image, list):
                image = [TF.hflip(img) for img in image]
            else:
                return TF.hflip(image)
        return image
class PairedRandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, prob=0.5):
        super(PairedRandomVerticalFlip, self).__init__(prob)
    def forward(self, image):
        if random.random() < self.p:
            if isinstance(image, list):
                image = [TF.vflip(img) for img in image]
            else:
                return TF.vflip(image)
        return image
class PairedRandomRotation(T.RandomRotation):
    def __init__(self, prob=0.5, degrees=(-90, 90)):
        super(PairedRandomRotation, self).__init__(degrees=degrees)
        self.p = prob
    def forward(self, image):
        if random.random() < self.p:
            degree = random.choice(self.degrees)
            if isinstance(image, list):
                image = [TF.rotate(img, degree) for img in image]
            else:
                return TF.rotate(image, degree)
        return image
class PairedNormalize(T.Normalize):
    def __init__(self, mean, std, inplace=True):
        super(PairedNormalize, self).__init__(mean=mean, std=std, inplace=inplace)
    def forward(self, image):
        if isinstance(image, list):
            image = [TF.normalize(img.float(), mean=self.mean, std=self.std, inplace=self.inplace) for img in image]
        else:
            return TF.normalize(image.float(), mean=self.mean, std=self.std, inplace=self.inplace)
        return image