import json
import joblib
import pandas as pd

from loguru import logger


def load_model(model_path, metadata_path):
    """
    Carrega o modelo treinado
    e os metadados da inferência.
    """

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])

    model_name = metadata["model_name"]

    logger.info(f"Modelo carregado: {model_name}")

    logger.info(f"Threshold carregado: {threshold:.2f}")

    return (model, model_name, threshold)


def predict(model, X, threshold):
    """
    Realiza inferência utilizando
    o threshold definido na validação.
    """

    y_proba = model.predict_proba(X)[:, 1]


    y_pred = (y_proba >= float(threshold)).astype(int)

    predictions = pd.DataFrame(
        {
            "prob_is_late": y_proba,
            "prediction": y_pred,
        },
        index=X.index,
    )

    return predictions