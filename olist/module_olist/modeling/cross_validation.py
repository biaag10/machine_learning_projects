from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)

from module_olist.config import FIGURES_DIR, REPORTS_DIR
from module_olist.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def summarize_cv(results):
    """
    Resume os resultados da validação cruzada.

    Para cada métrica, calcula:
    - média e desvio padrão dos folds de VALIDAÇÃO;
    - média dos folds de TREINO;
    - gap = treino - validação (diagnóstico de overfitting).

    Gap grande  -> o modelo memoriza o treino (overfitting).
    Treino e validação baixos -> o modelo é simples demais (underfitting).
    Desvio padrão grande -> o desempenho depende muito de quais dados caem
    em cada fold (modelo pouco robusto).
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

        val_values = results[f"test_{metric}"]
        train_values = results[f"train_{metric}"]

        summary.append(
            {
                "metric": metric,
                "mean": val_values.mean(),
                "std": val_values.std(),
                "train_mean": train_values.mean(),
                "gap": train_values.mean() - val_values.mean(),
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

        logger.info(
            f"CROSS VALIDATION - {name}\n"
            f"{summary.round(4).to_string(index=False)}"
        )

        # Diagnóstico de overfitting pelo PR-AUC (métrica principal)
        pr_auc_gap = summary.loc[
            summary["metric"] == "pr_auc", "gap"
        ].iloc[0]

        if pr_auc_gap > 0.10:
            logger.warning(
                f"[{name}] Possível overfitting: PR-AUC treino - validação = "
                f"{pr_auc_gap:.3f}"
            )
        else:
            logger.info(
                f"[{name}] Gap treino-validação no PR-AUC: {pr_auc_gap:.3f} "
                f"(sem sinal forte de overfitting)"
            )

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

            "pr_auc_std": (
                results[
                    "test_pr_auc"
                ].std()
            ),

            "pr_auc_gap": pr_auc_gap,
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
    # RELATÓRIOS DA VALIDAÇÃO
    # =========================================================

    save_cv_report(cv_results)

    best_pipeline = pipelines[best_model_name]

    compare_holdout_vs_cv(
        pipeline=best_pipeline,
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        cv_scores=cv_results[best_model_name]["results"]["test_pr_auc"],
    )

    plot_learning_curve(
        pipeline=best_pipeline,
        model_name=best_model_name,
        X_train=X_train,
        y_train=y_train,
        cv=kf,
    )

    # =========================================================
    # RETORNO
    # =========================================================

    return (
        best_model_name,
        best_threshold,
    )


# =============================================================
# FUNÇÕES DE APOIO À VALIDAÇÃO
# =============================================================


def save_cv_report(cv_results, output_path=REPORTS_DIR / "cv_results.csv"):
    """
    Salva a comparação dos modelos na validação cruzada:
    média e desvio dos folds, gap de overfitting e threshold escolhido.
    """

    rows = []

    for name, result in cv_results.items():

        summary = result["summary"].set_index("metric")

        rows.append(
            {
                "model": name,
                "roc_auc_mean": summary.loc["roc_auc", "mean"],
                "pr_auc_mean": summary.loc["pr_auc", "mean"],
                "pr_auc_std": summary.loc["pr_auc", "std"],
                "pr_auc_train": summary.loc["pr_auc", "train_mean"],
                "pr_auc_gap": summary.loc["pr_auc", "gap"],
                "threshold": result["threshold"],
                "precision_oof": result["precision"],
                "recall_oof": result["recall"],
                "f1_oof": result["f1_oof"],
            }
        )

    report = (
        pd.DataFrame(rows)
        .sort_values("f1_oof", ascending=False)
        .round(4)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    logger.info(f"Comparação dos modelos:\n{report.to_string(index=False)}")
    logger.success(f"Relatório da validação cruzada salvo em: {output_path}")

    return report


def compare_holdout_vs_cv(
    pipeline,
    model_name,
    X_train,
    y_train,
    cv_scores,
    n_repeats=5,
    output_path=REPORTS_DIR / "holdout_vs_cv.csv",
):
    """
    Compara validação SEM cross-validation (um único split treino/validação)
    com a validação COM cross-validation.

    Repetimos o holdout com sementes diferentes: se o resultado muda muito
    de uma semente para outra, uma única divisão não é confiável para
    escolher modelo e threshold. A cross-validation usa todos os dados
    para validar e entrega média +/- desvio.
    """

    holdout_scores = []

    for seed in range(n_repeats):

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=seed,
        )

        model = clone(pipeline)
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_val)[:, 1]
        holdout_scores.append(average_precision_score(y_val, y_proba))

    comparison = pd.DataFrame(
        {
            "estrategia": [
                f"holdout (1 split, {n_repeats} sementes)",
                f"cross-validation ({len(cv_scores)} folds)",
            ],
            "pr_auc_media": [np.mean(holdout_scores), np.mean(cv_scores)],
            "pr_auc_min": [np.min(holdout_scores), np.min(cv_scores)],
            "pr_auc_max": [np.max(holdout_scores), np.max(cv_scores)],
        }
    ).round(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    logger.info(
        f"[{model_name}] Holdout x Cross-Validation (PR-AUC):\n"
        f"{comparison.to_string(index=False)}"
    )
    logger.info(
        f"Com um único split, o PR-AUC poderia ser qualquer valor entre "
        f"{np.min(holdout_scores):.3f} e {np.max(holdout_scores):.3f}, "
        f"dependendo apenas da sorte na divisão."
    )

    return comparison


def plot_learning_curve(
    pipeline,
    model_name,
    X_train,
    y_train,
    cv,
    output_path=FIGURES_DIR / "learning_curve.png",
):
    """
    Curva de aprendizado: desempenho no treino e na validação
    conforme aumentamos a quantidade de dados de treino.

    - Curvas distantes  -> overfitting (mais dados ou menos complexidade ajudam)
    - Curvas próximas e baixas -> underfitting (modelo simples demais)
    - Curvas próximas e estáveis -> boa generalização
    """

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="average_precision",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_sizes, train_mean, marker="o", label="Treino")
    ax.plot(train_sizes, val_mean, marker="o", label="Validação (CV)")
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )

    ax.set(
        xlabel="Quantidade de pedidos no treino",
        ylabel="PR-AUC",
        title=f"Curva de aprendizado - {model_name}",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(
        f"[{model_name}] Curva de aprendizado: PR-AUC treino={train_mean[-1]:.3f} "
        f"| validação={val_mean[-1]:.3f} com todos os dados de treino."
    )
    logger.success(f"Curva de aprendizado salva em: {output_path}")
