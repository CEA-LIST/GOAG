import torch
import torch.nn as nn

##########################################################################################################################
## BPS
##########################################################################################################################

class BPSEncoder(nn.Module):
    def __init__(self,
                 layers_size=[2, 128, 256, 512],
                 latent_size=128):
        super(BPSEncoder, self).__init__()
        
        self.encoder = nn.Sequential()

        layers = []
        for i in range(len(layers_size) - 1):
            layers.append(nn.Conv1d(layers_size[i], layers_size[i + 1], kernel_size=3))
            layers.append(nn.BatchNorm1d(layers_size[i+1]))
            layers.append(nn.ReLU())
            nn.init.xavier_normal_(layers[-3].weight)

        layers.pop()        # remove last ReLU

        self.encoder = nn.Sequential(*layers)

        self.latent_enc_mu = nn.Linear(layers_size[-1], latent_size)
        self.latent_enc_logvar = nn.Linear(layers_size[-1], latent_size)

    def forward(self, x, y):
        """
        :param x: data (B, x_dim)
        :param y: condition (B, y_dim)
        :return mu, logvar: (B, latent_size)
        """
        x0 = torch.stack((y, x), dim=2)                                 # (B, y_dim + x_dim, 2)
        x = x0.permute(0, 2, 1)
        x = self.encoder(x)                                             # (B, layers_size[-1], (y_dim + x_dim) - 2 * (len(layers_size) - 1)) => no padding, (len(layers_size) - 1) Conv1d with kernel_size=3
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(x.size(0), -1)
        return self.latent_enc_mu(x), self.latent_enc_logvar(x)         # (B, latent_size), (B, latent_size)

