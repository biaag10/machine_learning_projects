import numpy as np

from loguru import logger

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold,
):
    """
    Avalia o modelo final utilizando o threshold
    definido durante a validação cruzada.
    """

    # Probabilidade da classe positiva
    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    # Converte probabilidade em classe
    # utilizando o threshold escolhido na CV
    y_pred = (
        y_proba >= threshold
    ).astype(int)

    # Métricas
    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    results = {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    logger.info(
        "========================================"
    )

    logger.info(
        "AVALIAÇÃO FINAL"
    )

    logger.info(
        f"Threshold: {threshold:.2f}"
    )

    logger.info(
        f"Accuracy: {accuracy:.3f}"
    )

    logger.info(
        f"Precision: {precision:.3f}"
    )

    logger.info(
        f"Recall: {recall:.3f}"
    )

    logger.info(
        f"F1: {f1:.3f}"
    )

    logger.info(
        f"ROC AUC: {roc_auc:.3f}"
    )

    logger.info(
        "========================================"
    )

    return results