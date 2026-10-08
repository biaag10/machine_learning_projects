from loguru import logger

from module_deep_learning.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from module_deep_learning.dataset import load_dataset
from module_deep_learning.features import create_features
from module_deep_learning.inference import run_inference
from module_deep_learning.modeling.evaluate import evaluate_model
from module_deep_learning.modeling.train import train_model
from module_deep_learning.plots import generate_plots
from module_deep_learning.split import split_data


def main():

    logger.info(
        "Iniciando preparação do dataset..."
    )

    X_train_raw, y_train_raw, X_test_raw, y_test = load_dataset(
        dataset_path=(
            RAW_DATA_DIR
            / "cifar10.npz"
        ),
    )

    # =============================================
    # FEATURES
    # Normalização dos pixels para [0, 1]
    # =============================================

    X_train_full = create_features(X_train_raw)
    X_test = create_features(X_test_raw)

    # =============================================
    # TRAIN / VALIDATION
    # =============================================

    X_train, X_val, y_train, y_val = (
        split_data(X_train_full, y_train_raw)
    )

    # =============================================
    # TREINAMENTO
    # =============================================

    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_path=(
            MODELS_DIR
            / "melhor_modelo_cifar10.keras"
        ),
        history_path=(
            REPORTS_DIR
            / "history.json"
        ),
    )

    # =============================================
    # TESTE FINAL
    # =============================================

    _, y_pred = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        metrics_path=(
            REPORTS_DIR
            / "metrics.json"
        ),
        predictions_path=(
            PROCESSED_DATA_DIR
            / "test_predictions.npz"
        ),
    )

    # =============================================
    # INFERÊNCIA DE EXEMPLO
    # =============================================

    run_inference(
        model=model,
        X=X_test_raw,
        y=y_test,
    )

    # =============================================
    # GRÁFICOS
    # Exemplos, curvas de treino e matriz de confusão
    # =============================================

    generate_plots(
        X=X_train_raw,
        y=y_train_raw,
        history=history,
        y_true=y_test,
        y_pred=y_pred,
    )

    logger.success(
        "Pipeline executado com sucesso."
    )


if __name__ == "__main__":
    main()