from torch.utils.data import Dataset
from models.blip import blip_decoder
import os
import csv
import glob

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cpu')


image_size = 384
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='large')
model.eval()
model = model.to(device)


class MyDataset(Dataset):
    def __init__(self, globdir):
        super().__init__()

        self.paths = list(glob.glob(globdir))
        self.len = len(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img = self.transform(img).unsqueeze(0)[0]
        return img, path


if __name__ == "__main__":
    import tqdm
    dataset = MyDataset(r"test\*")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=2)
    outcsv = []
    for imgs, paths in tqdm.tqdm(dataloader):
        imgs = imgs.to(device)
        captions = model.generate(imgs, sample=False, num_beams=3, max_length=60, min_length=20)

        for path, caption in zip(paths, captions):
            print(caption, os.path.basename(path))
            outcsv.append((caption, path))

        with open("result.csv", "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(outcsv)
