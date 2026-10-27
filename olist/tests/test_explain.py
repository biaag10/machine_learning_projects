import numpy as np
import pandas as pd

from module_olist.features import CATEGORICAL_FEATURES, FEATURES
from module_olist.modeling.interpret import (
    aggregate_shap_by_feature,
    calculate_shap_values,
    create_explainer,
    map_to_original_feature,
    prepare_data_for_shap,
)
from module_olist.pipeline import create_xgboost_pipeline


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
            "customer_state": rng.choice(["SP", "BA", "AL"], n),
        }
    )[FEATURES]
    y = pd.Series(((X["promised_days"] < 10) | (X["customer_state"] == "AL")).astype(int))
    return X, y


def test_nome_transformado_volta_para_feature_original():
    assert map_to_original_feature("numeric__promised_days", CATEGORICAL_FEATURES) == "promised_days"
    assert map_to_original_feature("categorical__customer_state_SP", CATEGORICAL_FEATURES) == "customer_state"
    assert map_to_original_feature("categorical__purchase_hour_10", CATEGORICAL_FEATURES) == "purchase_hour"


def test_shap_explica_exatamente_o_modelo_do_pipeline():
    """
    O SHAP é calculado na saída densa do pré-processador. O modelo precisa dar a
    MESMA previsão nesse formato e no pipeline completo; senão o SHAP estaria
    explicando um modelo diferente do que faz as previsões.
    """
    X, y = make_training_data()
    pipeline = create_xgboost_pipeline().fit(X, y)

    X_transformed = prepare_data_for_shap(pipeline, X)

    proba_pipeline = pipeline.predict_proba(X)[:, 1]
    proba_shap_input = pipeline.named_steps["model"].predict_proba(X_transformed.values)[:, 1]

    np.testing.assert_allclose(proba_pipeline, proba_shap_input, atol=1e-6)


def test_agregacao_preserva_a_soma_do_shap():
    """Somar as colunas One-Hot não muda a contribuição total de cada pedido."""
    X, y = make_training_data()
    pipeline = create_xgboost_pipeline().fit(X, y)

    X_transformed = prepare_data_for_shap(pipeline, X)
    shap_values = calculate_shap_values(create_explainer(pipeline), X_transformed)
    aggregated = aggregate_shap_by_feature(shap_values, X, CATEGORICAL_FEATURES)

    assert aggregated.feature_names == FEATURES
    np.testing.assert_allclose(
        aggregated.values.sum(axis=1), shap_values.values.sum(axis=1), rtol=1e-5, atol=1e-5
    )
