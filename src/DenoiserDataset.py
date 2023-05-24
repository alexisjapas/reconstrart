from torch.utils.data import Dataset
from torchvision.io.image import read_image
import os


class DenoiserDataset(Dataset):
    def __init__(self, data_path, noise, transform=None, val=False, split_ratio=1):
        super().__init__()
        extensions = [".png", ".jpg", ".jpeg"]
        self.paths = [
            os.path.join(data_path, p)
            for p in os.listdir(data_path)
            if os.path.splitext(p)[1] in extensions
        ]
        split_idx = int(split_ratio * len(self.paths))
        self.paths = self.paths[split_idx:] if val else self.paths[:split_idx]
        self.noise = noise
        self.transform = transform

    def __getitem__(self, idx):
        hr = read_image(self.paths[idx]).float()

        hr = (hr - hr.min()) / (hr.max() - hr.min())

        if self.transform:
            hr = self.transform(hr)

        ilr = self.noise(hr)

        return ilr, hr

    def __len__(self):
        return len(self.paths)
