from torchvision.transforms import transforms as T
from torchvision import io
from Generator import LESRCNNGenerator
import cv2 as cv
import torch
device = torch.device('cuda:0')

t = T.Compose([
    T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.])
])
network = LESRCNNGenerator().to(device)
network.load_state_dict(torch.load('trained_lesrcnn.pt'))
network.eval()

image_path = "./val_images/d.png"

with torch.no_grad():
    image = t(io.read_image(image_path, io.ImageReadMode.RGB).to(device).float().unsqueeze(dim=0))  # [1, C, H, W]
    _, _, h, w = image.shape
    output_image = torch.round(torch.clip(network(image).permute(0, 2, 3, 1) * 255.0, min=0, max=255))[0] \
        .cpu().numpy().astype('uint8')

output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
cv.imwrite("val_generated.png", output_image)
