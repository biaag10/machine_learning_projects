import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def evaluate_model(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        y_proba = model.predict_proba(X_test)[:,1]

        best_threshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None


        for threshold in np.arange(0.05,0.5,0.01):

            y_pred = (
                y_proba >= threshold
            ).astype(int)


            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )


            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall


        roc_auc = roc_auc_score(
            y_test,
            y_proba
        )


        results[name] = {
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "roc_auc": roc_auc
        }


        logger.info(f"Modelo: {name}")
        logger.info(f"Threshold: {best_threshold:.2f}")
        logger.info(f"F1: {best_f1:.3f}")
        logger.info(f"ROC AUC: {roc_auc:.3f}")


    return results