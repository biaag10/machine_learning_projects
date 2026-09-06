from loguru import logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
    threshold,
):
    """
    Realiza a avaliação final do modelo
    no conjunto de teste.
    """

    # Probabilidade da classe positiva
    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    # Aplica o threshold definido na validação
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

    pr_auc = average_precision_score(
        y_test,
        y_proba,
    )

    # Resultados Finais
    logger.info("=" * 60)

    logger.success(
        f"MODELO FINAL: {model_name}"
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
        f"ROC-AUC: {roc_auc:.3f}"
    )

    logger.info(
        f"PR-AUC: {pr_auc:.3f}"
    )