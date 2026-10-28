# TREINAMENTO DO DDPM
#
# A cada iteração: sorteia t, sorteia o ruído ε, cria x_t e treina a rede para acertar ε:
#
#   L = || ε − ε_θ(x_t, t, y) ||²      (MSE: aprendizado supervisionado, o alvo é o ruído conhecido)
#
# Em 10% das vezes a classe é trocada por "sem condição" (necessário para o CFG) e mantemos
# uma EMA (média móvel) dos pesos, que gera imagens mais limpas.
#
# Uso:  python -m module_diffusion_models.modeling.train --epochs 10

import copy
from pathlib import Path
import time

from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F
import typer

from module_diffusion_models.config import (
    CH,
    EMA_DECAY,
    EPOCHS,
    FIGURES_DIR,
    LR,
    MODEL_PATH,
    P_UNCOND,
    SEED,
    get_device,
)
from module_diffusion_models.dataset import get_dataloader, load_mnist
from module_diffusion_models.diffusion import NoiseSchedule
from module_diffusion_models.modeling.unet import UNet, count_parameters
from module_diffusion_models.plots import plot_loss

app = typer.Typer()


def train_step(model, ema, schedule, x, y, opt, scaler=None, p_uncond=P_UNCOND):
    """Um passo de treino. Retorna o valor da loss (MSE do ruído)."""
    device = x.device
    # esconde a classe em p_uncond das amostras (Classifier-Free Guidance)
    hide = torch.rand(len(y), device=device) < p_uncond
    y = torch.where(hide, torch.full_like(y, model.null_class), y)
    t = torch.randint(0, schedule.T, (len(x),), device=device)
    noise = torch.randn_like(x)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        loss = F.mse_loss(model(schedule.q_sample(x, t, noise), t, y), noise)

    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():  # EMA dos pesos
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - EMA_DECAY)

    return loss.item()


def train(train_dl, epochs: int = EPOCHS, ch: int = CH, device: str = "cpu"):
    """Treina a U-Net do zero. Retorna o modelo EMA e a loss de cada iteração."""
    torch.manual_seed(SEED)
    schedule = NoiseSchedule(device=device)
    model = UNet(ch=ch).to(device)
    logger.info(
        f"U-Net com {count_parameters(model) / 1e6:.2f} M parâmetros | dispositivo: {device}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched_lr = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=epochs * len(train_dl), pct_start=0.05
    )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    losses, t_start = [], time.time()
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for x, y in train_dl:
            loss = train_step(model, ema, schedule, x.to(device), y.to(device), opt, scaler)
            sched_lr.step()
            losses.append(loss)
            ep_loss += loss
        logger.info(
            f"época {epoch + 1:2d}/{epochs} | loss médio {ep_loss / len(train_dl):.4f} | "
            f"{time.time() - t_start:.0f}s"
        )
    return ema, losses


@app.command()
def main(epochs: int = EPOCHS, model_path: Path = MODEL_PATH):
    device = get_device()
    train_dl = get_dataloader(load_mnist(train=True))
    ema, losses = train(train_dl, epochs=epochs, device=device)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema.state_dict(), model_path)
    plot_loss(losses, FIGURES_DIR / "05_curva_loss.png")
    logger.success(f"Modelo salvo em {model_path}")


if __name__ == "__main__":
    app()
