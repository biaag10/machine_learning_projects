import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from module_olist.features import FEATURES
from module_olist.modeling.cross_validation import (
    compare_holdout_vs_cv,
    find_best_threshold,
    plot_learning_curve,
    summarize_cv,
)
from module_olist.pipeline import create_lightgbm_pipeline

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def make_training_data(n=300, seed=0):
    """Base sintética pequena com as mesmas colunas do modelo real."""
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


def test_summarize_cv_calcula_gap_de_overfitting():
    results = {}
    for metric in METRICS:
        results[f"train_{metric}"] = np.array([0.9, 0.9])
        results[f"test_{metric}"] = np.array([0.6, 0.8])

    summary = summarize_cv(results).set_index("metric")

    assert summary.loc["pr_auc", "mean"] == 0.7
    assert summary.loc["pr_auc", "train_mean"] == 0.9
    assert round(summary.loc["pr_auc", "gap"], 2) == 0.2


def test_threshold_maximiza_f1():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.35, 0.9])

    result = find_best_threshold(y_true, y_proba)

    # Com threshold entre 0.31 e 0.35 o modelo acerta tudo: F1 = 1
    assert result["f1"] == 1.0
    assert 0.3 < result["threshold"] <= 0.35


def test_holdout_e_curva_de_aprendizado_geram_arquivos(tmp_path):
    X, y = make_training_data(n=300)
    pipeline = create_lightgbm_pipeline().set_params(model__verbose=-1)

    comparison = compare_holdout_vs_cv(
        pipeline=pipeline,
        model_name="LightGBM",
        X_train=X,
        y_train=y,
        cv_scores=np.array([0.5, 0.6, 0.7]),
        n_repeats=2,
        output_path=tmp_path / "holdout.csv",
    )
    assert len(comparison) == 2
    assert (tmp_path / "holdout.csv").exists()

    plot_learning_curve(
        pipeline=pipeline,
        model_name="LightGBM",
        X_train=X,
        y_train=y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        output_path=tmp_path / "lc.png",
    )
    assert (tmp_path / "lc.png").exists()
