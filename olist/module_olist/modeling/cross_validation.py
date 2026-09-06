from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import pandas as pd

from loguru import logger

from module_olist.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def summarize_cv(results):
    """
    Resume os resultados da validação cruzada.

    Para cada métrica, calcula:
    - média dos folds;
    - desvio padrão dos folds.
    """

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = []

    for metric in metrics:

        values = results[f"test_{metric}"]

        summary.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
            }
        )

    return pd.DataFrame(summary)


def find_best_threshold(y_true, y_proba):
    """
    Encontra o threshold que maximiza o F1-score.

    O threshold é testado entre 0.01 e 0.99.
    """

    best_threshold = None
    best_f1 = -1

    best_accuracy = None
    best_precision = None
    best_recall = None

    logger.info(
        "Testando thresholds de 0.01 até 0.99..."
    )

    for threshold in np.arange(0.01, 1.00, 0.01):

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if f1 > best_f1:

            best_threshold = threshold

            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return {
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def cross_validate_models(
    X_train,
    y_train,
):
    """
    Executa validação cruzada dos modelos.

    Para cada modelo:

    - executa Cross Validation;
    - calcula métricas dos folds;
    - gera probabilidades Out-of-Fold;
    - encontra o melhor threshold pelo F1;
    - compara os modelos pelo F1 OOF otimizado.

    Retorna:

    - nome do melhor modelo;
    - melhor threshold.
    """

    # =========================================================
    # PIPELINES
    # =========================================================

    pipelines = {
        "Gradient Boosting": (
            create_gradient_boosting_pipeline()
        ),

        "XGBoost": (
            create_xgboost_pipeline()
        ),

        "LightGBM": (
            create_lightgbm_pipeline()
        ),
    }

    # =========================================================
    # CROSS VALIDATION
    # =========================================================

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # =========================================================
    # MÉTRICAS
    # =========================================================

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_results = {}

    total_models = len(pipelines)

    # =========================================================
    # EXECUÇÃO DOS MODELOS
    # =========================================================

    for model_index, (name, pipeline) in enumerate(
        pipelines.items(),
        start=1,
    ):

        logger.info(
            "=" * 60
        )

        logger.info(
            f"MODELO {model_index}/{total_models}: {name}"
        )

        logger.info(
            "Iniciando Cross Validation com 5 folds..."
        )

        # =====================================================
        # CROSS VALIDATE
        # =====================================================

        logger.info(
            f"[{name}] Treinando os 5 folds..."
        )

        results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        logger.success(
            f"[{name}] Cross Validation concluído."
        )

        # =====================================================
        # RESUMO
        # =====================================================

        summary = summarize_cv(
            results
        )

        print("\n")
        print("=" * 60)
        print(
            f"CROSS VALIDATION - {name}"
        )
        print("=" * 60)
        print(summary.to_string(index=False))
        print("=" * 60)

        # =====================================================
        # OUT-OF-FOLD
        # =====================================================

        logger.info(
            f"[{name}] Calculando probabilidades "
            f"Out-of-Fold..."
        )

        y_proba_oof = cross_val_predict(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        logger.success(
            f"[{name}] Probabilidades Out-of-Fold calculadas."
        )

        # =====================================================
        # THRESHOLD
        # =====================================================

        logger.info(
            f"[{name}] Procurando melhor threshold..."
        )

        threshold_results = find_best_threshold(
            y_true=y_train,
            y_proba=y_proba_oof,
        )

        # =====================================================
        # RESULTADOS
        # =====================================================

        cv_results[name] = {
            "results": results,

            "summary": summary,

            "threshold": (
                threshold_results[
                    "threshold"
                ]
            ),

            "accuracy": (
                threshold_results[
                    "accuracy"
                ]
            ),

            "precision": (
                threshold_results[
                    "precision"
                ]
            ),

            "recall": (
                threshold_results[
                    "recall"
                ]
            ),

            "f1_oof": (
                threshold_results[
                    "f1"
                ]
            ),

            "pr_auc": (
                results[
                    "test_pr_auc"
                ].mean()
            ),
        }

        # =====================================================
        # RESULTADO DO MODELO
        # =====================================================

        logger.success(
            f"THRESHOLD OTIMIZADO - {name}"
        )

        logger.info(
            f"Threshold: "
            f"{threshold_results['threshold']:.2f}"
        )

        logger.info(
            f"Accuracy: "
            f"{threshold_results['accuracy']:.3f}"
        )

        logger.info(
            f"Precision: "
            f"{threshold_results['precision']:.3f}"
        )

        logger.info(
            f"Recall: "
            f"{threshold_results['recall']:.3f}"
        )

        logger.info(
            f"F1 OOF: "
            f"{threshold_results['f1']:.3f}"
        )

        logger.info(
            f"PR-AUC médio: "
            f"{results['test_pr_auc'].mean():.3f}"
        )

        logger.success(
            f"Modelo {name} finalizado "
            f"({model_index}/{total_models})."
        )

    # =========================================================
    # SELEÇÃO DO MELHOR MODELO
    # =========================================================

    logger.info(
        "=" * 60
    )

    logger.info(
        "Comparando os modelos..."
    )

    best_model_name = max(
        cv_results,
        key=lambda name: (
            cv_results[name]["f1_oof"]
        ),
    )

    best_results = cv_results[
        best_model_name
    ]

    best_threshold = (
        best_results["threshold"]
    )

    best_accuracy = (
        best_results["accuracy"]
    )

    best_precision = (
        best_results["precision"]
    )

    best_recall = (
        best_results["recall"]
    )

    best_f1 = (
        best_results["f1_oof"]
    )

    best_pr_auc = (
        best_results["pr_auc"]
    )

    # =========================================================
    # RESULTADO FINAL
    # =========================================================

    logger.success(
        "MODELO SELECIONADO"
    )

    logger.info(
        f"Modelo: {best_model_name}"
    )

    logger.info(
        f"Threshold: {best_threshold:.2f}"
    )

    logger.info(
        f"Accuracy OOF: {best_accuracy:.3f}"
    )

    logger.info(
        f"Precision OOF: {best_precision:.3f}"
    )

    logger.info(
        f"Recall OOF: {best_recall:.3f}"
    )

    logger.info(
        f"F1 OOF: {best_f1:.3f}"
    )

    logger.info(
        f"PR-AUC médio CV: {best_pr_auc:.3f}"
    )

    logger.success(
        "Cross Validation concluído."
    )

    # =========================================================
    # RETORNO
    # =========================================================

    return (
        best_model_name,
        best_threshold,
    )