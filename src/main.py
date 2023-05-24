import torch
from torch.nn import Sequential
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import threading
import os
from lightning.pytorch.loggers import TensorBoardLogger

from LitDenoiser import LitDenoiser
from DenoiserDataset import DenoiserDataset
from VDSR import VDSR


def launchTensorBoard():
    os.system("tensorboard --logdir=.")


# monitoring
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()
logger = TensorBoardLogger(save_dir=".")

# optimizations
torch.set_float32_matmul_precision("medium")

# data
data_path = "/home/qosu/data/munch_paintings"
split_ratio = 0.99
noise = Sequential(transforms.GaussianBlur(kernel_size=(7, 7), sigma=(2.0, 2.0)))
# transform = Sequential(transforms.RandomCrop(64))
train_dataset = DenoiserDataset(
    data_path=data_path,
    noise=noise,
    transform=transforms.RandomCrop(82),
    val=False,
    split_ratio=split_ratio,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
)
val_dataset = DenoiserDataset(
    data_path=data_path,
    noise=noise,
    transform=transforms.RandomCrop(82),
    val=True,
    split_ratio=split_ratio,
)
val_dataloader = DataLoader(val_dataset, num_workers=8)

# model
model = VDSR(20, 3, 64)
denoiser = LitDenoiser(denoiser=model)

# training
trainer = pl.Trainer(max_epochs=80, logger=logger)
trainer.fit(
    model=denoiser, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)
