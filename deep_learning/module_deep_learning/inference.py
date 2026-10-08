from loguru import logger

from module_deep_learning.config import CLASS_NAMES, FIGURES_DIR
from module_deep_learning.dataset import load_dataset
from module_deep_learning.features import create_features
from module_deep_learning.modeling.predict import load_model, predict
from module_deep_learning.plots import plot_samples, save_figure


def run_inference(model=None, X=None, y=None, n_samples=9):

    logger.info("Executando inferência de exemplo...")

    # X deve ser a imagem original (uint8), para conseguir plotar
    if X is None or y is None:
        _, _, X, y = load_dataset()

    if model is None:
        model = load_model()

    X_sample = X[:n_samples]
    y_sample = y[:n_samples]

    probs, y_pred = predict(
        model,
        create_features(X_sample),
    )

    for i in range(n_samples):
        logger.info(
            f"Imagem {i}: "
            f"real={CLASS_NAMES[y_sample[i]]} | "
            f"previsto={CLASS_NAMES[y_pred[i]]} "
            f"({probs[i, y_pred[i]]:.1%})"
        )

    save_figure(
        plot_samples(
            X_sample,
            y_sample,
            y_pred,
            title="Predições do modelo para as primeiras 9 imagens de teste.",
        ),
        FIGURES_DIR / "predicoes_teste.png",
    )

    return y_pred


def main():
    run_inference()


if __name__ == "__main__":
    main()