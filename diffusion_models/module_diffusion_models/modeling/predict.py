# PROCESSO REVERSO: GERANDO IMAGENS
#
# - DDPM (Ho et al., 2020): 1000 passos estocásticos. Qualidade de referência, porém lento.
# - DDIM (Song et al., 2021): determinístico e pula passos (ex.: 50). Mesmo modelo, sem retreinar.
# - Classifier-Free Guidance: ε̂ = ε_incond + w·(ε_cond − ε_incond)
#
# Uso:  python -m module_diffusion_models.modeling.predict --digito 7 --passos 50 --guidance 3

from pathlib import Path

from loguru import logger
import torch
import typer

from module_diffusion_models.config import (
    CH,
    DDIM_STEPS,
    FIGURES_DIR,
    GUIDANCE,
    IMG_SIZE,
    MODEL_PATH,
    get_device,
)
from module_diffusion_models.diffusion import NoiseSchedule
from module_diffusion_models.modeling.unet import UNet
from module_diffusion_models.plots import save_grid

app = typer.Typer()


def load_model(path: Path = MODEL_PATH, ch: int = CH, device: str = "cpu") -> UNet:
    """Carrega os pesos (EMA) do modelo treinado."""
    model = UNet(ch=ch).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval().requires_grad_(False)


@torch.no_grad()
def predict_eps(model, x, t_int, y, guidance):
    """CFG: w=0 -> incondicional, w=1 -> só condicional, w>1 -> reforça a classe."""
    b = x.shape[0]
    t = torch.full((b,), t_int, device=x.device, dtype=torch.long)
    x_in, t_in = torch.cat([x, x]), torch.cat([t, t])
    y_in = torch.cat([y, torch.full_like(y, model.null_class)])
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(x.device.type == "cuda")
    ):
        out = model(x_in, t_in, y_in).float()
    eps_c, eps_u = out.chunk(2)
    return eps_u + guidance * (eps_c - eps_u)


def _initial_noise(n, device, seed):
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    return torch.randn(n, 1, IMG_SIZE, IMG_SIZE, device=device, generator=g), g


@torch.no_grad()
def sample_ddpm(model, schedule: NoiseSchedule, y, guidance=GUIDANCE, save_every=None, seed=None):
    """Processo REVERSO estocástico: percorre os T passos, de x_T (ruído) até x_0."""
    device = y.device
    x, g = _initial_noise(len(y), device, seed)
    frames = []
    for t in reversed(range(schedule.T)):
        eps = predict_eps(model, x, t, y, guidance)
        ab = schedule.alphas_bar[t]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)  # estimativa da imagem limpa
        mean = (schedule.alphas_bar_prev[t].sqrt() * schedule.betas[t] / (1 - ab)) * x0 + (
            schedule.alphas[t].sqrt() * (1 - schedule.alphas_bar_prev[t]) / (1 - ab)
        ) * x  # média da posterior
        if t > 0:
            noise = torch.randn(x.shape, device=device, generator=g)
            x = mean + schedule.posterior_var[t].sqrt() * noise
        else:
            x = mean
        if save_every and (t % save_every == 0 or t == schedule.T - 1):
            frames.append((t, x.detach().cpu()))
    return x.clamp(-1, 1), frames


@torch.no_grad()
def sample_ddim(
    model,
    schedule: NoiseSchedule,
    y,
    steps=DDIM_STEPS,
    guidance=GUIDANCE,
    save_every=None,
    seed=None,
):
    """Processo REVERSO determinístico: pula passos (ex.: 1000 -> 50)."""
    device = y.device
    x, _ = _initial_noise(len(y), device, seed)
    ts = torch.linspace(schedule.T - 1, 0, steps).long().tolist()
    frames = []
    for i, t in enumerate(ts):
        eps = predict_eps(model, x, t, y, guidance)
        ab = schedule.alphas_bar[t]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()  # ε coerente com o x0 recortado
        ab_prev = (
            schedule.alphas_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        )
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        if save_every and (i % save_every == 0):
            frames.append((t, x.detach().cpu()))
    frames.append((0, x.detach().cpu()))
    return x.clamp(-1, 1), frames


@app.command()
def main(
    digito: int = typer.Option(-1, help="Dígito a gerar (0-9). -1 gera todos os dígitos."),
    quantidade: int = typer.Option(8, help="Amostras por dígito."),
    passos: int = typer.Option(DDIM_STEPS, help="Passos do DDIM (use 1000 com --ddpm)."),
    guidance: float = typer.Option(GUIDANCE, help="Intensidade do guidance (w)."),
    ddpm: bool = typer.Option(False, help="Usa o DDPM com 1000 passos em vez do DDIM."),
    seed: int = 0,
    model_path: Path = MODEL_PATH,
):
    device = get_device()
    model = load_model(model_path, device=device)
    schedule = NoiseSchedule(device=device)

    digitos = list(range(10)) if digito < 0 else [digito]
    y = torch.tensor(digitos, device=device).repeat_interleave(quantidade)

    if ddpm:
        imgs, _ = sample_ddpm(model, schedule, y, guidance=guidance, seed=seed)
        nome = f"geradas_ddpm1000_w{guidance:g}"
    else:
        imgs, _ = sample_ddim(model, schedule, y, steps=passos, guidance=guidance, seed=seed)
        nome = f"geradas_ddim{passos}_w{guidance:g}"

    # cada coluna é um dígito
    grid = imgs.view(len(digitos), quantidade, 1, IMG_SIZE, IMG_SIZE).transpose(0, 1)
    path = save_grid(
        grid.reshape(-1, 1, IMG_SIZE, IMG_SIZE), FIGURES_DIR / f"{nome}.png", nrow=len(digitos)
    )
    logger.success(f"{len(y)} imagens geradas em {path}")


if __name__ == "__main__":
    app()
