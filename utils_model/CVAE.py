import torch
import lightning as L

from utils_model.Encoder import BPSEncoder
from utils_model.Decoder import BPSDecoder
from utils_model.LossCVAE import LossCVAE

class CVAE(L.LightningModule):
    def __init__(self, cfg):
        super(CVAE, self).__init__()

        self.cfg = cfg
        self.latent_size = cfg['latent_size']

        self.N = torch.distributions.Normal(0, 1)

        self.loss_criterion = LossCVAE(beta_range=self.cfg['beta_range'],
                                        beta_cycle_start=self.cfg['beta_cycle_start'],
                                        beta_cycle_length=self.cfg['beta_cycle_length'],
                                        sigmoid_scale=self.cfg['sigmoid_scale'],
                                        is_cyclique=self.cfg['is_cyclique'],
                                        batchsize=self.cfg['batch_size'], 
                                        beta_fixed=self.cfg['beta_fixed'])  

        self.train_logs = {
            'loss': [],
            'recon': [],
            'kld': [],
            'error': [],
            'square_error_w_attn': []
        }
        self.validate_logs = {
            'loss': [],
            'recon': [],
            'kld': [],
            'error': [],
            'square_error_w_attn': []
        }
            
        self.encoder = BPSEncoder(layers_size=self.cfg['layers_size'],
                                    latent_size=self.cfg['latent_size'])

        self.decoder = BPSDecoder(x_dim=self.cfg['out_dim'],
                                  y_dim=self.cfg['in_dim'],
                                  n_neurons=self.cfg['layers_size'][-1],
                                  latent_size=self.cfg['latent_size'])


    def forward(self, x, y):
        """
        :param x: (B, N)
        :param y: condition = bps (B, N)
        :return:
        """
        means, logvars = self.encoder(x, y)                                                 # (B, latent_size), (B, latent_size)
        z_latent_code = self.reparameterize(means=means, logvars=logvars)                   # (B, latent_size)
        cp_hat = self.decoder(y, z_latent_code)                                             # (B, N)
        return cp_hat, means, logvars, z_latent_code

    def inference(self, y, z_latent_code):
        """
        :param y: condition = bps (B, N)
        :param z_latent_code: (B, latent_size)
        :return cp_hat: (B, N)
        """
        cp_hat = self.decoder(y, z_latent_code)
        return cp_hat

    def reparameterize(self, means, logvars):
        """
        :param means: (B, latent_size) 
        :param logvars: (B, latent_size)
        :return z_latent_code: (B, latent_size) 
        """
        self.N.loc = self.N.loc.to(means.device)
        self.N.scale = self.N.scale.to(means.device)
        std = torch.exp(0.5 * logvars)
        return means + std * self.N.sample(means.shape)

    def training_step(self, batch, batch_idx):
        _, _, _, _, bps, cp_gt, _ = batch
        cp_hat, means, logvars, _ = self(cp_gt, bps)
        loss, recon, kld, error, square_error = self.loss_criterion(means, logvars, bps, cp_gt, cp_hat)
        self.train_logs['loss'].append(loss.item())
        self.train_logs['recon'].append(recon.item())
        self.train_logs['kld'].append(kld.item())
        self.train_logs['error'].append(error.item())
        self.train_logs['square_error_w_attn'].append(square_error.item())

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, _, bps, cp_gt, _ = batch
        cp_hat, means, logvars, _ = self(cp_gt, bps)
        loss, recon, kld, error, square_error = self.loss_criterion(means, logvars, bps, cp_gt, cp_hat)
        
        self.validate_logs['loss'].append(loss.item())
        self.validate_logs['recon'].append(recon.item())
        self.validate_logs['kld'].append(kld.item())
        self.validate_logs['error'].append(error.item())
        self.validate_logs['square_error_w_attn'].append(square_error.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.cfg['learning_rate']))
        return optimizer

    def on_train_epoch_end(self):
        self.loss_criterion.apply_iter()
        self.logger.experiment.add_scalar('beta', self.loss_criterion.beta, self.current_epoch)

        self.log('val_loss', sum(self.validate_logs['loss']) / len(self.validate_logs['loss']), logger=False, sync_dist=True)

        for key in self.train_logs.keys():
            avg_train_loss = sum(self.train_logs[key]) / len(self.train_logs[key])
            avg_validate_loss = sum(self.validate_logs[key]) / len(self.validate_logs[key])

            self.logger.experiment.add_scalars(key, {'train': avg_train_loss, 'validate': avg_validate_loss}, self.current_epoch)

            # Reset logs
            self.train_logs[key] = []
            self.validate_logs[key] = []
