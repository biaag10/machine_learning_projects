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

    # Testa diferentes thresholds
    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):

        # Converte probabilidades em classes
        y_pred = (
            y_proba >= threshold
        ).astype(int)

        # Calcula as métricas
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

        # Verifica se encontrou um F1 melhor
        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

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
    - Executa Cross Validation;
    - Calcula métricas dos folds;
    - Gera probabilidades Out-of-Fold;
    - Encontra o melhor threshold pelo F1;
    - Compara os modelos pelo F1 OOF otimizado.

    Retorna:
    - nome do melhor modelo;
    - melhor threshold.
    """

    # ============================================================
    # PIPELINES
    # ============================================================

    pipelines = {
        "Gradient Boosting":
            create_gradient_boosting_pipeline(),

        "XGBoost":
            create_xgboost_pipeline(),

        "LightGBM":
            create_lightgbm_pipeline(),
    }


    # ============================================================
    # ESTRATÉGIA DE CROSS VALIDATION
    # ============================================================

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


    # ============================================================
    # MÉTRICAS
    # ============================================================

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }


    # Guarda os resultados dos modelos
    cv_results = {}


    # ============================================================
    # CROSS VALIDATION
    # ============================================================

    for name, pipeline in pipelines.items():

        print("\n")
        print("=" * 60)
        print(f"MODELO: {name}")
        print("=" * 60)

        logger.info(
            f"[{name}] Iniciando Cross Validation..."
        )


        # ========================================================
        # 1 - MÉTRICAS DOS FOLDS
        # ========================================================

        print(
            f"[{name}] [1/3] "
            f"Iniciando Cross Validation..."
        )

        results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=False,
        )

        print(
            f"[{name}] [1/3] "
            f"Cross Validation concluído!"
        )

        logger.success(
            f"[{name}] Cross Validation concluído."
        )


        # ========================================================
        # RESUMO DAS MÉTRICAS
        # ========================================================

        print(
            f"[{name}] Calculando resumo das métricas..."
        )

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


        # ========================================================
        # 2 - PROBABILIDADES OUT-OF-FOLD
        # ========================================================

        print(
            f"\n[{name}] [2/3] "
            f"Gerando probabilidades Out-of-Fold..."
        )

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
        )[:, 1]

        print(
            f"[{name}] [2/3] "
            f"Out-of-Fold concluído!"
        )

        logger.success(
            f"[{name}] Probabilidades OOF concluídas."
        )


        # ========================================================
        # 3 - MELHOR THRESHOLD
        # ========================================================

        print(
            f"\n[{name}] [3/3] "
            f"Procurando melhor threshold..."
        )

        logger.info(
            f"[{name}] Testando thresholds de "
            f"0.01 até 0.99..."
        )

        threshold_results = find_best_threshold(
            y_true=y_train,
            y_proba=y_proba_oof,
        )

        print(
            f"[{name}] [3/3] "
            f"Threshold encontrado: "
            f"{threshold_results['threshold']:.2f}"
        )

        logger.success(
            f"[{name}] Threshold otimizado."
        )


        # ========================================================
        # GUARDA OS RESULTADOS
        # ========================================================

        cv_results[name] = {

            "results": results,

            "summary": summary,

            "threshold": (
                threshold_results["threshold"]
            ),

            "accuracy": (
                threshold_results["accuracy"]
            ),

            "precision": (
                threshold_results["precision"]
            ),

            "recall": (
                threshold_results["recall"]
            ),

            "f1_oof": (
                threshold_results["f1"]
            ),

            "pr_auc": (
                results[
                    "test_pr_auc"
                ].mean()
            ),
        }


        # ========================================================
        # RESULTADO COM THRESHOLD OTIMIZADO
        # ========================================================

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


        # ========================================================
        # MODELO CONCLUÍDO
        # ========================================================

        print("\n")
        print("=" * 60)
        print(
            f"{name} CONCLUÍDO!"
        )
        print("=" * 60)


    # ============================================================
    # ESCOLHA DO MELHOR MODELO
    # ============================================================

    logger.info(
        "Comparando os modelos pelo F1 OOF..."
    )

    best_model_name = max(
        cv_results,
        key=lambda name: (
            cv_results[name]["f1_oof"]
        ),
    )


    # Recupera os resultados do vencedor
    best_results = cv_results[
        best_model_name
    ]

    best_threshold = best_results[
        "threshold"
    ]

    best_accuracy = best_results[
        "accuracy"
    ]

    best_precision = best_results[
        "precision"
    ]

    best_recall = best_results[
        "recall"
    ]

    best_f1 = best_results[
        "f1_oof"
    ]

    best_pr_auc = best_results[
        "pr_auc"
    ]


    # ============================================================
    # RESULTADO FINAL DA SELEÇÃO
    # ============================================================

    print("\n")
    print("=" * 60)
    print("MODELO SELECIONADO")
    print("=" * 60)

    print(
        f"Modelo: {best_model_name}"
    )

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Accuracy OOF: {best_accuracy:.3f}"
    )

    print(
        f"Precision OOF: {best_precision:.3f}"
    )

    print(
        f"Recall OOF: {best_recall:.3f}"
    )

    print(
        f"F1 OOF: {best_f1:.3f}"
    )

    print(
        f"PR-AUC médio CV: {best_pr_auc:.3f}"
    )

    print("=" * 60)


    logger.success(
        f"Melhor modelo: "
        f"{best_model_name}"
    )

    logger.info(
        f"F1 OOF otimizado: "
        f"{best_f1:.3f}"
    )

    logger.success(
        "Cross Validation concluído."
    )


    # ============================================================
    # RETORNO
    # ============================================================

    return (
        best_model_name,
        best_threshold,
    )