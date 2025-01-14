import torch
from torch import nn
import torch.nn.functional as F


class EncoderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_hid_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hid_layers)]
        )
        self.mu_out = nn.Linear(hidden_dim, latent_dim)
        self.logvar_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.mu_out(x), self.logvar_out(x)


class DecoderVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, n_hid_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hid_layers)],
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.layers(x)


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, n_hid_layers):
        super().__init__()
        self.encoder = EncoderVAE(input_dim, hidden_dim, latent_dim, n_hid_layers)
        self.decoder = DecoderVAE(output_dim, hidden_dim, latent_dim, n_hid_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def _reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.randn_like(mu) * std + mu

    def sample(self, mu, logvar):
        z = self._reparametrization_trick(mu, logvar)
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        x_hat = self.sample(mu, logvar)
        return x_hat, mu, logvar

    @staticmethod
    def loss_bce(x, x_hat, mu, logvar, kl_weight=0.5):
        rec_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') 
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar - 1)
        return rec_loss + kl_weight * kl_loss, rec_loss, kl_loss

    @staticmethod
    def loss_mse(x, x_hat, mu, logvar, kl_weight=0.5):
        rec_loss = torch.sum((x_hat - x)** 2) 
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar - 1)
        return rec_loss + kl_weight * kl_loss, rec_loss, kl_loss