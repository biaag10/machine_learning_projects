# ARQUITETURA: U-NET CONDICIONADA
#
# Entrada: imagem com ruído x_t (1x32x32), passo t e classe y.  Saída: ruído previsto ε̂ (1x32x32).
#
#   Encoder 32 -> 16 -> 8 (contexto)  ·  Gargalo com self-attention  ·  Decoder 8 -> 16 -> 32 (detalhe)
#   Skip connections ligam os dois lados; o embedding (tempo + classe) entra em todos os blocos.

import math

import torch
from torch import nn
import torch.nn.functional as F

from module_diffusion_models.config import CH, NUM_CLASSES


class SinusoidalEmb(nn.Module):
    """Transforma o passo t (inteiro) em um vetor, igual ao positional encoding do Transformer."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Bloco residual que recebe também o embedding (tempo + classe)."""

    def __init__(self, cin, cout, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = (
            h + self.emb(F.silu(emb))[:, :, None, None]
        )  # "em que passo estou" e "qual dígito quero"
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention espacial (usado na resolução 8x8)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)  # (B, H*W, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """U-Net que prevê o ruído ε̂ a partir de (x_t, t, y)."""

    def __init__(self, ch: int = CH, num_classes: int = NUM_CLASSES):
        super().__init__()
        emb_dim = ch * 4
        self.null_class = num_classes  # índice extra = "sem condição" (para CFG)
        self.time_emb = nn.Sequential(
            SinusoidalEmb(ch), nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        # ---- Encoder (descida) ----
        self.d1 = ResBlock(ch, ch, emb_dim)
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # 32 -> 16
        self.d2 = ResBlock(ch, ch * 2, emb_dim)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)  # 16 -> 8
        self.d3 = ResBlock(ch * 2, ch * 2, emb_dim)
        self.a3 = Attention(ch * 2)
        # ---- Gargalo ----
        self.m1 = ResBlock(ch * 2, ch * 2, emb_dim)
        self.ma = Attention(ch * 2)
        self.m2 = ResBlock(ch * 2, ch * 2, emb_dim)
        # ---- Decoder (subida) com skip connections ----
        self.u3 = ResBlock(ch * 4, ch * 2, emb_dim)
        self.ua3 = Attention(ch * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 8 -> 16
        self.u2 = ResBlock(ch * 4, ch * 2, emb_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 16 -> 32
        self.u1 = ResBlock(ch * 3, ch, emb_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t, y):
        emb = self.time_emb(t) + self.class_emb(y)
        h0 = self.inc(x)
        s1 = self.d1(h0, emb)  # ch   x 32x32
        s2 = self.d2(self.down1(s1), emb)  # 2ch  x 16x16
        s3 = self.a3(self.d3(self.down2(s2), emb))  # 2ch  x 8x8
        h = self.m2(self.ma(self.m1(s3, emb)), emb)
        h = self.ua3(self.u3(torch.cat([h, s3], 1), emb))
        h = self.u2(torch.cat([self.up2(h), s2], 1), emb)
        h = self.u1(torch.cat([self.up1(h), s1], 1), emb)
        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
