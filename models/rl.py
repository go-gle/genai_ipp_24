from ddpm import TimeEmb, DownSamplingBlock, ResBlock
from torch import nn
import numpy as np
import torch


class EnergyModelMNIST(nn.Module):
    def __init__(self, emb_channels, init_dim):
        super().__init__()
        self.init_proj = nn.Conv2d(1, init_dim, kernel_size=3, padding=1)
        self.time_emb = TimeEmb(emb_channels)

        self.res1 = ResBlock(init_dim, 1 * init_dim, emb_channels)
        self.down1 = DownSamplingBlock(1 * init_dim, 1 * init_dim, emb_channels)
        self.res2 = ResBlock(1 * init_dim, 2 * init_dim, emb_channels)
        self.down2 = DownSamplingBlock(2 * init_dim, 2 * init_dim, emb_channels)
        self.res3 = ResBlock(2 * init_dim, 3 * init_dim, emb_channels)
        self.down3 = DownSamplingBlock(3 * init_dim, 3 * init_dim, emb_channels)

        self.res_last = ResBlock(3 * init_dim, 3 * init_dim,  emb_channels)
        self.norm = nn.GroupNorm(8, 3 * init_dim)
        self.selu = nn.SELU()
        self.out = nn.Conv2d(3 * init_dim, 1, 7)

    def forward(self, x, t):
        h = self.init_proj(x)
        t_emb = self.time_emb(t)
        h = self.res1(h, t_emb)
        _, h = self.down1(h, t_emb)
        h = self.res2(h, t_emb)
        _, t = self.down2(h, t_emb)
        h = self.res3(h, t_emb)
        _, h = self.down3(h,t_emb)
        h = self.res_last(h, t_emb)
        return self.out(self.selu(self.norm(h))).squeeze()


class RLSampler():
    def __init__(self, model, n_steps=6, langevin_steps=30, device='cpu'):
        self.n_steps = n_steps
        self.device = device
        self.langevin_steps = langevin_steps
        self.langevin_b_square = 2e-4
        # Recovery mask
        self.is_recovery = torch.ones(self.n_steps + 1, device=device)
        self.is_recovery[-1] = 0
        self.model = model.to(device)

        betas = np.append(np.linspace(0.0001, 0.02, 1_000), 1)
        sqrt_alphas = np.sqrt(1. - betas)
        idx = np.concatenate(
            [np.arange(n_steps) * (1000 // ((n_steps - 1) * 2)), [999]]
            )
        a_s = np.concatenate(
            [[np.prod(sqrt_alphas[: idx[0] + 1])],
        np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
        sigma = np.sqrt(1 - a_s ** 2)

        self.sigmas = torch.FloatTensor(sigma).to(self.device)
        self.a_s = torch.FloatTensor(a_s).to(self.device)
        self.a_s_cum = torch.FloatTensor(np.cumprod(a_s)).to(self.device)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum ** 2)

    @staticmethod
    def gather(const_to_gather, t):
        return const_to_gather.gather(-1, t).reshape(-1, 1, 1, 1)

    def q_xt_x0(self, x0, t):
        mean = self.gather(self.a_s_cum, t) * x0
        var = self.gather(self.sigmas_cum, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, std = self.q_xt_x0(x0, t)
        return mean + std * eps

    def q_sample_pairs(self, x0, t):
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t)
        x_t_next = self.gather(self.a_s, t + 1) * x_t + self.gather(self.sigmas, t + 1) * eps
        return x_t, x_t_next

    def training_losses(self, x_pos, x_neg, t):
        a_s = self.gather(self.a_s_prev, t + 1)
        y_pos = a_s * x_pos
        y_neg = a_s * x_neg
        pos_f = self.model(y_pos, t).sum()
        neg_f = self.model(y_neg, t).sum()
        loss = - (pos_f - neg_f)
        loss_scale = 1.0 / (self.sigmas[t + 1] / self.sigmas[1])
        loss = loss_scale * loss
        return loss.mean()

    def log_prob(self, y, t, tilde_x, b0, sigma, is_recovery):
        # print(f'model sampling {y}, {t}')
        logits = self.model(y, t)
        # print(f'logits {logits}')
        # print(b0.flatten())
        # print(sigma)
        return logits.sum() / b0.flatten() - torch.sum((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery, dim=[1, 2, 3])

    def grad_f(self, y, t, tilde_x, b0, sigma, is_recovery):
        log_p_y = self.log_prob(y, t, tilde_x, b0, sigma, is_recovery)
        grad_y = torch.autograd.grad(log_p_y.sum(), [y], retain_graph=True)[0]
        return grad_y, log_p_y

    def p_sample_langevin(self, tilde_x, t):
        """
        Langevin sampling function
        """
        sigma = self.gather(self.sigmas, t + 1)
        sigma_cum = self.gather(self.sigmas_cum, t)
        is_recovery = self.gather(self.is_recovery, t + 1)
        a_s = self.gather(self.a_s_prev, t + 1)

        c_t_square = sigma_cum / self.sigmas_cum[0]
        step_size_square = c_t_square * self.langevin_b_square * sigma ** 2

        y = tilde_x.clone().detach()
        y.requires_grad = True

        grad_y, log_p_y = self.grad_f(y, t, tilde_x, step_size_square, sigma, is_recovery)

        for _ in range(self.langevin_steps):
            noise = torch.randn_like(y)
            y_new = y + 0.5 * step_size_square * grad_y + torch.sqrt(step_size_square) * noise
            grad_y_new, log_p_y_new = self.grad_f(y_new, t, tilde_x, step_size_square, sigma, is_recovery)
            y, grad_y = y_new, grad_y_new
        x = y / a_s
        return x

    def p_sample_given_xt(self, noise):
            num = noise.shape[0]
            x_neg_t = noise
            x_neg = torch.zeros([self.n_steps, num, 1, 28, 28], device=self.device)
            x_neg = torch.cat([x_neg, torch.unsqueeze(noise, axis=0)], dim=0)

            for t in range(self.n_steps - 1, -1, -1):
                t_v = torch.tensor([t] * num).to(self.device)
                x_neg_t  = self.p_sample_langevin(x_neg_t, t_v)
                x_neg_t = torch.reshape(x_neg_t, [num, 1, 28, 28])
                insert_mask = t == torch.arange(self.n_steps + 1, device=self.device)
                insert_mask = torch.reshape(insert_mask, [-1, *([1] * len(noise.shape))])
                x_neg = insert_mask * torch.unsqueeze(x_neg_t, axis=0) + (~ insert_mask) * x_neg
            return x_neg
