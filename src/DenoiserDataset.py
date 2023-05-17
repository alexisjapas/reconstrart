from torch.utils.data import Dataset
import os


class DenoiserDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.paths = [
            os.path.join(data_path, p)
            for p in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, p))
        ]
        self.transform = transform

    def __getitem__(self, idx):
        pass  # TODO

    def __len__(self):
        return len(self.paths)
