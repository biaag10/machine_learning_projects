import json

import numpy as np
import pandas as pd

from module_olist.features import FEATURES
from module_olist.modeling.evaluate import evaluate_model
from module_olist.modeling.predict import load_model, predict
from module_olist.modeling.train import train_model


def make_training_data(n=400, seed=0):
    """Base sintética com as mesmas colunas do modelo real."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "promised_days": rng.uniform(2, 40, n),
            "item_count": rng.integers(1, 4, n),
            "seller_count": rng.integers(1, 3, n),
            "total_price": rng.uniform(10, 500, n),
            "total_freight": rng.uniform(5, 60, n),
            "approval_hours": rng.uniform(0, 48, n),
            "avg_item_price": rng.uniform(10, 300, n),
            "freight_ratio": rng.uniform(0, 1, n),
            "is_short_deadline": rng.integers(0, 2, n),
            "is_weekend_purchase": rng.integers(0, 2, n),
            "purchase_hour": rng.integers(0, 24, n),
            "purchase_weekday": rng.integers(0, 7, n),
            "purchase_month": rng.integers(1, 13, n),
            "customer_state": rng.choice(["SP", "BA", "RJ"], n),
        }
    )[FEATURES]
    y = pd.Series((X["promised_days"] < 10).astype(int))
    return X, y


def test_treino_salva_modelo_e_metadados(tmp_path):
    X, y = make_training_data()

    train_model(
        model_name="LightGBM",
        threshold=0.3,
        X_train=X,
        y_train=y,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
    )

    model, model_name, threshold = load_model(
        tmp_path / "model.joblib", tmp_path / "metadata.json"
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert model_name == "LightGBM"
    assert threshold == 0.3
    assert metadata["features"] == FEATURES

    predictions = predict(model, X.head(5), threshold)
    assert set(predictions.columns) == {"prob_is_late", "prediction"}
    assert predictions["prob_is_late"].between(0, 1).all()


def test_avaliacao_salva_metricas_e_compara_com_baseline(tmp_path):
    X, y = make_training_data()

    model = train_model(
        model_name="XGBoost",
        threshold=0.5,
        X_train=X,
        y_train=y,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
    )

    report = evaluate_model(
        model=model,
        model_name="XGBoost",
        X_test=X,
        y_test=y,
        threshold=0.5,
        X_train=X,
        y_train=y,
        metrics_path=tmp_path / "metrics.json",
        figure_path=tmp_path / "cm.png",
    )

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cm.png").exists()

    # O baseline nunca encontra atrasos: recall zero
    assert report["baseline_most_frequent"]["recall"] == 0

    cm = report["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(y)
