import json

import joblib
from loguru import logger
import numpy as np
import pandas as pd


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


def validate_features(X, features):
    """
    Garante que a entrada da inferência tem as mesmas colunas do treino.
    Sem isso, um erro de preparação dos dados só apareceria lá dentro do
    pipeline, com uma mensagem difícil de entender.
    """

    missing = [column for column in features if column not in X.columns]

    if missing:
        raise ValueError(f"Colunas ausentes para a inferência: {missing}")

    return X[features]


def classify_risk(y_proba, threshold):
    """
    Traduz a probabilidade em uma faixa de risco para a operação:

    - "alto":  probabilidade >= 2 x threshold  -> acompanhar com prioridade
    - "médio": threshold <= probabilidade < 2 x threshold -> monitorar
    - "baixo": probabilidade < threshold -> fluxo normal
    """

    return np.select(
        [y_proba >= 2 * threshold, y_proba >= threshold],
        ["alto", "médio"],
        default="baixo",
    )


def predict(model, X, threshold):
    """
    Realiza inferência utilizando
    o threshold definido na validação.
    """

    # Probabilidade de atraso (classe positiva)
    y_proba = model.predict_proba(X)[:, 1]

    # Decisão: acima do threshold, o pedido é sinalizado como risco de atraso
    y_pred = (y_proba >= float(threshold)).astype(int)

    predictions = pd.DataFrame(
        {
            "prob_is_late": y_proba,
            "prediction": y_pred,
            "risk_level": classify_risk(y_proba, float(threshold)),
        },
        index=X.index,
    )

    return predictions