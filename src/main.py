import torch
from torch.nn import Sequential
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from LitDenoiser import LitDenoiser
from DenoiserDataset import DenoiserDataset
from VDSR import VDSR


# optimizations
torch.set_float32_matmul_precision("medium")

# data
data_path = "/home/qosu/data/munch_paintings"
noise = Sequential(transforms.GaussianBlur(kernel_size=(3, 3)))
transform = Sequential(transforms.RandomCrop(64))
dataset = DenoiserDataset(data_path=data_path, noise=noise, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
)

# model
model = VDSR(2, 3, 8)
denoiser = LitDenoiser(denoiser=model)

# training
trainer = pl.Trainer(max_epochs=1, log_every_n_steps=10)
trainer.fit(model=denoiser, train_dataloaders=dataloader)
