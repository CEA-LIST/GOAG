import torch
import torch.nn as nn

from utils_model.NetworkBlocks import ResBlock

##########################################################################################################################
## BPS
##########################################################################################################################

class BPSDecoder(nn.Module):
    def __init__(self,
                 x_dim=4096,
                 y_dim=4096,
                 n_neurons=512,
                 latent_size=128):
        super(BPSDecoder, self).__init__()

        self.bn = nn.BatchNorm1d(y_dim)
        self.block1 = ResBlock(y_dim + latent_size, n_neurons)
        self.block2 = ResBlock(y_dim + latent_size + n_neurons, n_neurons)

        self.fc = nn.Linear(n_neurons, x_dim)
        self.activation = nn.Sigmoid()

    def forward(self, y, z):
        """
        :param y: condition (B, y_dim)
        :param z: (B, latent_size)
        :return x_hat: (B, x_dim)
        """

        y0 = self.bn(y)
        x0 = torch.cat((y0, z), dim=1)
        x = self.block1(x0)
        x = self.block2(torch.cat((x0, x), dim=1))

        return self.activation(self.fc(x))
