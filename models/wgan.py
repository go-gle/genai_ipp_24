from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=64, hid_dims=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hid_dims = hid_dims
        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim , self.hid_dims, 3, 2),
            nn.BatchNorm2d(self.hid_dims),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hid_dims, self.hid_dims * 2, 4, 1),
            nn.BatchNorm2d(self.hid_dims * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hid_dims * 2, self.hid_dims, 3, 2),
            nn.BatchNorm2d(self.hid_dims),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hid_dims, 1, 4, 2),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.proj(z)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.out(z)
        return z


class Critic(nn.Module):
    def __init__(self, latent_dim=64, hid_dims=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hid_dims = hid_dims
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hid_dims, kernel_size=4, stride=2),
            nn.BatchNorm2d(hid_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hid_dims, out_channels=hid_dims * 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(2 * hid_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2 * hid_dims, out_channels=hid_dims, kernel_size=4, stride=2),
            nn.Flatten(1),
            nn.Linear(self.hid_dims, self.hid_dims),
            nn.BatchNorm1d(hid_dims),
            nn.ReLU(),
            nn.Linear(self.hid_dims, 1)
            )


    def forward(self, x):
        x = self.out(x)
        return x