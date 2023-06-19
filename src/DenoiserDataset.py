from torch.utils.data import Dataset
from torchvision.io.image import read_image


class DenoiserDataset(Dataset):
    def __init__(self, paths: list, noise, transform=None):
        super().__init__()
        self.paths = paths
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
