import torch.nn as nn


class VDSR(nn.Module):
    def __init__(self, n_layers=20, n_channels=64, skip_connection=True):
        super().__init__()
        assert n_layers >= 2
        self.n_layers = n_layers - 2
        self.skip_connection = skip_connection
        self.input = nn.Conv2d(1, n_channels, 3, padding='same')
        self.hidden_conv = nn.Conv2d(n_channels, n_channels, 3, padding='same')
        self.relu = nn.ReLU()
        self.output = nn.Conv2d(n_layers, 1, 3, padding='same')

    def forward(self, x):
        ilr = x.copy()
        x = self.input(x)
        for _ in range(self.n_layers):
            x = self.hidden_conv(x)
            x = self.relu(x)
        x = self.output(x)
        if self.skip_connection:
            x += ilr
        return x
