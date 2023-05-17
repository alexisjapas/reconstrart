import lightning.pytorch as pl
from torch import nn, optim


class LitDenoiser(pl.LightningModule):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.denoiser(x)
        loss = nn.functional.mse_loss(z, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
