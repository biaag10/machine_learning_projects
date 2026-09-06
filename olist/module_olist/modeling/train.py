import json
import joblib

from loguru import logger

from module_olist.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)

def create_selected_model(
    model_name,
):
    """
    Cria o pipeline correspondente
    ao modelo selecionado.
    """

    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline,
        "XGBoost": create_xgboost_pipeline,
        "LightGBM": create_lightgbm_pipeline,
    }

    if model_name not in pipelines:
        raise ValueError(
            f"Modelo desconhecido: {model_name}"
        )

    return pipelines[model_name]()


def train_model(
    model_name,
    threshold,
    X_train,
    y_train,
    model_path,
    metadata_path,
):
    """
    Treina o modelo selecionado utilizando
    todo o conjunto de treinamento
    e salva modelo e metadados.
    """

    logger.info(
        f"Treinando modelo final: {model_name}"
    )

    # -------------------------------------------------
    # Cria somente o modelo vencedor
    # -------------------------------------------------

    model = create_selected_model(
        model_name
    )

    # -------------------------------------------------
    # Treina com TODO o conjunto de treino
    # -------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------
    # Salva pipeline completo
    # -------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    logger.success(
        f"Modelo salvo em: {model_path}"
    )

    # -------------------------------------------------
    # Salva metadados
    # -------------------------------------------------

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "selection_metric": "pr_auc",
        "threshold_metric": "f1",
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    logger.success(
        f"Metadados salvos em: {metadata_path}"
    )

    return model