import json

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from module_deep_learning.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from module_deep_learning.dataset import load_dataset


def save_figure(fig, figure_path):

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(f"Figura salva em: {figure_path}")

    return figure_path


def plot_samples(X, y, y_pred=None, title="Exemplos da base CIFAR-10."):

    fig = plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])

        label = f"Real: {CLASS_NAMES[y[i]]}"
        if y_pred is not None:
            label += f"\nPredição: {CLASS_NAMES[y_pred[i]]}"

        plt.title(label)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_history(history, metric, title):

    fig = plt.figure(figsize=(8, 5))

    plt.plot(history[metric], label="Treino")
    plt.plot(history[f"val_{metric}"], label="Validação")

    plt.title(f"{title} por época")
    plt.xlabel("Épocas")
    plt.ylabel(title)
    plt.legend()

    return fig


def plot_confusion_matrix(y_true, y_pred):

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=NUM_CLASSES,
    ).numpy()

    # Normalizada por linha: % de cada classe real
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("Matriz de confusão (conjunto de teste)")

    # Contagem e percentual em cada célula
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            cor = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.0%}",
                ha="center",
                va="center",
                color=cor,
                fontsize=8,
            )

    fig.tight_layout()

    return fig


def generate_plots(X=None, y=None, history=None, y_true=None, y_pred=None):

    logger.info("Gerando gráficos...")

    if X is None or y is None:
        X, y, _, _ = load_dataset()

    if history is None:
        history = json.loads(
            (REPORTS_DIR / "history.json").read_text(encoding="utf-8")
        )

    if y_true is None or y_pred is None:
        with np.load(PROCESSED_DATA_DIR / "test_predictions.npz") as data:
            y_true = data["y_true"]
            y_pred = data["y_pred"]

    paths = [
        save_figure(
            plot_samples(X, y),
            FIGURES_DIR / "exemplos_cifar10.png",
        ),
        save_figure(
            plot_history(history, "accuracy", "Acurácia"),
            FIGURES_DIR / "acuracia_por_epoca.png",
        ),
        save_figure(
            plot_history(history, "loss", "Loss"),
            FIGURES_DIR / "loss_por_epoca.png",
        ),
        save_figure(
            plot_confusion_matrix(y_true, y_pred),
            FIGURES_DIR / "matriz_confusao.png",
        ),
    ]

    logger.success("Gráficos gerados com sucesso.")

    return paths


def main():
    generate_plots()


if __name__ == "__main__":
    main()