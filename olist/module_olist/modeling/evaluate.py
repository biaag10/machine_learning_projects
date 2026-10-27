import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from module_olist.config import FIGURES_DIR, REPORTS_DIR


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcula as métricas de classificação.

    - accuracy, precision, recall e F1 usam a CLASSE prevista (após o threshold);
    - ROC-AUC e PR-AUC usam a PROBABILIDADE (não dependem do threshold).
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def evaluate_baseline(X_train, y_train, X_test, y_test):
    """
    Baseline "ingênuo": sempre prevê a classe majoritária (no prazo).

    Serve para mostrar que a acurácia engana em bases desbalanceadas:
    ele acerta ~92% sem encontrar NENHUM pedido atrasado.
    """

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)[:, 1]

    return calculate_metrics(y_test, y_pred, y_proba)


def plot_confusion_matrix(y_true, y_pred, model_name, output_path: Path):
    """Salva a matriz de confusão do conjunto de teste."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No prazo", "Atrasado"],
        cmap="Blues",
        values_format=",d",
        ax=ax,
    )

    ax.set(
        title=f"Matriz de confusão - {model_name} (teste)",
        xlabel="Previsto",
        ylabel="Real",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Matriz de confusão salva em: {output_path}")


def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
    threshold,
    X_train=None,
    y_train=None,
    metrics_path=REPORTS_DIR / "test_metrics.json",
    figure_path=FIGURES_DIR / "confusion_matrix.png",
):
    """
    Realiza a avaliação final do modelo no conjunto de teste.

    O teste fica "trancado" até aqui: ele não foi usado para escolher
    modelo nem threshold, então esta é uma estimativa imparcial.
    """

    # Probabilidade da classe positiva (atraso)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplica o threshold definido na validação cruzada
    y_pred = (y_proba >= threshold).astype(int)

    metrics = calculate_metrics(y_test, y_pred, y_proba)

    # Matriz de confusão: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Resultados Finais
    logger.info("=" * 60)
    logger.success(f"MODELO FINAL: {model_name}")
    logger.info(f"Threshold: {threshold:.2f}")

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.3f}")

    logger.info(
        f"Matriz de confusão -> TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}"
    )
    logger.info(
        f"O modelo encontrou {tp:,} de {tp + fn:,} pedidos atrasados "
        f"e gerou {fp:,} alertas falsos."
    )

    report = {
        "model_name": model_name,
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }

    # Comparação com o baseline (sempre "no prazo")
    if X_train is not None and y_train is not None:
        baseline_metrics = evaluate_baseline(X_train, y_train, X_test, y_test)

        report["baseline_most_frequent"] = {
            name: round(float(value), 4) for name, value in baseline_metrics.items()
        }

        logger.info(
            f"Baseline (sempre 'no prazo'): accuracy={baseline_metrics['accuracy']:.3f} "
            f"| recall={baseline_metrics['recall']:.3f} | F1={baseline_metrics['f1']:.3f}"
        )

    # Salva métricas e gráfico
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    logger.success(f"Métricas de teste salvas em: {metrics_path}")

    plot_confusion_matrix(y_test, y_pred, model_name, figure_path)

    return report