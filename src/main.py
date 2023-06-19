import torch
from torch.nn import Sequential
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import threading
import os
from lightning.pytorch.loggers import TensorBoardLogger
from math import floor

from LitDenoiser import LitDenoiser
from DenoiserDataset import DenoiserDataset
from VDSR import VDSR


def launchTensorBoard():
    os.system("tensorboard --logdir=.")


# monitoring
# t = threading.Thread(target=launchTensorBoard, args=([]))
# t.start()
logger = TensorBoardLogger(save_dir=".")

# optimizations
torch.set_float32_matmul_precision("medium")

# data
data_path = "/home/qosu/data/munch_paintings"
extensions = [".png", ".jpg", ".jpeg"]
paths = [
    os.path.join(data_path, p)
    for p in os.listdir(data_path)
    if os.path.splitext(p)[1] in extensions
]
noise = Sequential(transforms.GaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0)))

train_paths = paths[: floor(0.7 * len(paths))]
train_ds = DenoiserDataset(
    paths=train_paths,
    noise=noise,
    transform=transforms.RandomCrop(82),
)
train_dl = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
)

valid_paths = paths[floor(0.7 * len(paths)) : floor(0.9 * len(paths))]
valid_ds = DenoiserDataset(
    paths=valid_paths,
    noise=noise,
    transform=transforms.RandomCrop(82),
)
valid_dl = DataLoader(valid_ds, num_workers=8)

test_paths = paths[floor(0.9 * len(paths)) :]
test_ds = DenoiserDataset(paths=test_paths, noise=noise)
test_dl = DataLoader(test_ds)

# model
model = VDSR(20, 3, 64)
denoiser = LitDenoiser(denoiser=model)

# training
trainer = pl.Trainer(max_epochs=8, logger=logger)
trainer.fit(model=denoiser, train_dataloaders=train_dl, val_dataloaders=valid_dl)

trainer.test(model=denoiser, dataloaders=test_dl)
