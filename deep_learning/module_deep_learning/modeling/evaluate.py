import json

from loguru import logger
import numpy as np
import tensorflow as tf

from module_deep_learning.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from module_deep_learning.dataset import load_dataset
from module_deep_learning.features import create_features
from module_deep_learning.modeling.predict import load_model, predict


def per_class_accuracy(y_true, y_pred):

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=NUM_CLASSES,
    ).numpy()

    acc = cm.diagonal() / cm.sum(axis=1)

    # Ordena da pior para a melhor classe
    return dict(
        sorted(
            zip(CLASS_NAMES, acc.tolist()),
            key=lambda item: item[1],
        )
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    metrics_path=REPORTS_DIR / "metrics.json",
    predictions_path=PROCESSED_DATA_DIR / "test_predictions.npz",
):

    logger.info("Avaliando no conjunto de teste...")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    probs, y_pred = predict(model, X_test)

    acc_por_classe = per_class_accuracy(y_test, y_pred)

    logger.success(
        f"Acurácia no teste: {test_acc * 100:.2f}% | "
        f"Loss no teste: {test_loss:.4f}"
    )

    logger.info("Acurácia por classe:")
    for nome, acc in acc_por_classe.items():
        logger.info(f"  {nome:<10} {acc * 100:.1f}%")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "per_class_accuracy": acc_por_classe,
    }

    # Salva métricas
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Salva predições (usadas pelos gráficos)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=probs,
    )

    logger.info(f"Métricas salvas em: {metrics_path}")
    logger.info(f"Predições salvas em: {predictions_path}")

    return metrics, y_pred


def main():

    _, _, X_test, y_test = load_dataset()

    X_test = create_features(X_test)

    model = load_model()

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()