# PROCESSO DIRETO (FORWARD) E AGENDAMENTO DE RUÍDO
#
# Adicionamos ruído gaussiano aos poucos, em T = 1000 passos, com variâncias β_t crescentes.
# Graças a uma propriedade da gaussiana, dá para pular direto de x_0 para qualquer x_t:
#
#   x_t = sqrt(ᾱ_t)·x_0 + sqrt(1 - ᾱ_t)·ε ,   ε ~ N(0, I) ,   ᾱ_t = Π_{s<=t} (1 - β_s)
#
# Esse processo não tem nada a aprender: quem aprende é a rede que faz o caminho inverso.

import torch

from module_diffusion_models.config import BETA_END, BETA_START, T


class NoiseSchedule:
    """Agendamento linear de ruído (Ho et al., 2020) e as grandezas derivadas dele."""

    def __init__(
        self,
        timesteps: int = T,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
        device: str = "cpu",
    ):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # fração de sinal que sobra em t
        self.alphas_bar_prev = torch.cat([torch.ones(1, device=device), self.alphas_bar[:-1]])
        # variância da posterior q(x_{t-1} | x_t, x_0), usada no DDPM
        self.posterior_var = self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Processo DIRETO: leva x_0 para x_t em um único passo."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alphas_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
