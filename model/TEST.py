import torch.nn as nn


class TESTModel(nn.Module):
    def __init__(self):
        super(TESTModel, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
