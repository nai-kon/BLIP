import time

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip_vqa import blip_vqa

device = torch.device('cpu')


def load_image(path, image_size, device):
    raw_image = Image.open(path).convert('RGB')
    raw_image = ImageOps.exif_transpose(raw_image)

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


if __name__ == "__main__":
    image_size = 480
    image = load_image(r"test.jpg", image_size, device)

    print("booting...")
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    print("ready")
    with torch.no_grad():
        while True:
            q = input()
            t1 = time.time()
            answer = model(image, q, train=False, inference='generate')
            t2 = time.time()

            print(f'ans({t2-t1:.1f} sec): ', answer)
