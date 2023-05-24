import lightning.pytorch as pl
import torch
from torchvision.utils import make_grid


class LitDenoiser(pl.LightningModule):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.denoiser(x)
        print(f"min: {z.min()} max: {z.max()}")
        loss = torch.nn.functional.mse_loss(z, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.denoiser(x)
        loss = torch.nn.functional.mse_loss(z, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.epoch_examples = [x[0], y[0], z[0]]

        return loss

    def on_validation_epoch_end(self):
        # log images
        x, y, z = self.epoch_examples
        hr = y  # (y - y.min()) / (y.max() - y.min())
        ilr = x  # (x - x.min()) / (x.max() - x.min())
        ilr = torch.flip(ilr, [2])
        out = z  # (z - z.min()) / (z.max() - z.min())
        diff = z - x
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        examples = make_grid([hr, ilr, out, diff])
        self.logger.experiment.add_image("examples", examples, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=20, gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
