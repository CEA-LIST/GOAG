import math
import torch
import torch.nn as nn


class LossCVAE(nn.Module):
    def __init__(self, batchsize, beta_cycle_start, beta_cycle_length,
                beta_range=[0.0001, 0.1], sigmoid_scale=20.0, is_cyclique=False, beta_fixed=None):
        super(LossCVAE, self).__init__()
        
        self.beta_range = [float(beta_range[0]), float(beta_range[1])]
        self.beta = self.beta_range[0]
        self.beta_cycle_start = beta_cycle_start
        self.cycle_length = beta_cycle_length
        self.sigmoid_scale = sigmoid_scale
        self.is_cyclique = is_cyclique
        self.beta_fixed = beta_fixed

        self.batchsize = batchsize

        self.iter_counter = 0

    def forward(self, means, logvars, bps, cp_gt, cp_hat):
        """
        :param means:
        :param logvars:
        :param cp_gt: B x N
        :param cp_hat: B x N
        :return:
        """
        # Loss KLD
        loss_kld = -0.5 *  torch.mean(1 + logvars - means.pow(2) - torch.exp(logvars))

        # L2 Loss
        error = torch.abs(cp_gt - cp_hat)
        # print("error: ", error.size())
        ret_error = error.clone().detach().mean()
        square_error = torch.square(error)
        # print("square_error: ", square_error.size())
        attention_weights = torch.exp(cp_gt * 3.0)
        # print("attention_weights: ", attention_weights.size())
        square_error = square_error.mul_(attention_weights)
        # print("square_error: ", square_error.size())
        mean_square_error = square_error.sum(dim=1) / attention_weights.sum(dim=1)
        # print("mean_square_error: ", mean_square_error.size())
        ret_square_error = square_error.clone().detach().mean()
        # square_error = square_error.mul(attention_weights)
        loss_recon = torch.sqrt(mean_square_error).mean()

        if loss_recon.isnan().any():
            print(f"recon: {loss_recon.item()}, error: {ret_error.item()}, square_error: {ret_square_error.item()}")
            data = {
                "cp_gt": cp_gt,
                "cp_hat": cp_hat,
                "bps": bps}
            print("Loss recon is NaN. Data saved.")
            torch.save(data, "loss_recon_nan.pt")
            
            import sys
            sys.exit()

        # Loss
        loss = self.beta * loss_kld + loss_recon
        return loss, loss_recon, loss_kld, ret_error, ret_square_error

    def apply_iter(self):
        if self.beta_fixed:
            self.beta = self.beta_fixed
        else:
            if self.iter_counter < self.beta_cycle_start:
                self.beta = self.beta_range[0]
            else:
                if self.is_cyclique:
                    phase = ((self.iter_counter - self.beta_cycle_start) % self.cycle_length) / self.cycle_length
                    self.beta = self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) / (1 + math.exp(-self.sigmoid_scale * (phase - 0.5)))    # Apply sigmoid function to smoothly transition in the beta_range
                else:
                    if self.iter_counter < self.cycle_length + self.beta_cycle_start:
                        phase = ((self.iter_counter - self.beta_cycle_start) % self.cycle_length) / self.cycle_length
                        self.beta = self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) / (1 + math.exp(-self.sigmoid_scale * (phase - 0.5)))
                    else:
                        self.beta = self.beta_range[1]
        # print(f"\n[LOSS] beta={self.beta}")
        self.iter_counter += 1

