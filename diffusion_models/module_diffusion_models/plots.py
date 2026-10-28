# FIGURAS: grades de imagens e curva de loss

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision


def to_grid(imgs, nrow: int = 10):
    """Lote de imagens em [-1, 1] -> grade (H, W, 3) em [0, 1]."""
    grid = torchvision.utils.make_grid((imgs.cpu() + 1) / 2, nrow=nrow, padding=2, pad_value=1)
    return grid.permute(1, 2, 0).clamp(0, 1).numpy()


def save_grid(imgs, path: Path, nrow: int = 10, title: str | None = None) -> Path:
    """Salva um lote de imagens como uma grade PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = to_grid(imgs, nrow)
    plt.figure(figsize=(10, 10 * grid.shape[0] / grid.shape[1]))
    plt.imshow(grid, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_loss(losses, path: Path) -> Path:
    """Curva de loss por iteração (escala log) com média móvel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k = max(1, min(100, len(losses) // 5))
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(7, 3))
    plt.plot(losses, alpha=0.25, label="por iteração")
    plt.plot(range(k - 1, len(losses)), smooth, label=f"média móvel ({k})")
    plt.yscale("log")
    plt.xlabel("iteração")
    plt.ylabel("MSE do ruído")
    plt.legend()
    plt.title("Treinamento")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return path
