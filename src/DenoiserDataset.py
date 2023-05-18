from torch.utils.data import Dataset
from torchvision.io.image import read_image
import os


class DenoiserDataset(Dataset):
    def __init__(self, data_path, noise, transform=None):
        super().__init__()
        extensions = [".png", ".jpg", ".jpeg"]
        self.paths = [
            os.path.join(data_path, p)
            for p in os.listdir(data_path)
            if os.path.splitext(p)[1] in extensions
        ]
        self.noise = noise
        self.transform = transform

    def __getitem__(self, idx):
        hr = read_image(self.paths[idx]).float()
        if self.transform:
            hr = self.transform(hr)

        inlr = self.noise(hr)

        return inlr, hr

    def __len__(self):
        return len(self.paths)
