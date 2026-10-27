import numpy as np
import pandas as pd

from module_olist.active_learning import (
    max_predicted_probability,
    run_active_learning,
    uncertainty_sampling,
)
from module_olist.features import FEATURES


class FakeModel:
    """Modelo falso que devolve probabilidades fixas, para testar a seleção."""

    def __init__(self, proba_late):
        self.proba_late = np.asarray(proba_late)

    def predict_proba(self, X):
        return np.column_stack([1 - self.proba_late, self.proba_late])


def make_training_data(n=600, seed=0):
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
    y = pd.Series((X["promised_days"] < 12).astype(int))
    return X, y


def test_confianca_e_a_maior_probabilidade_entre_as_classes():
    model = FakeModel([0.1, 0.5, 0.9])
    confidence = max_predicted_probability(model, pd.DataFrame(index=range(3)))
    np.testing.assert_allclose(confidence, [0.9, 0.5, 0.9])


def test_incerteza_escolhe_os_pedidos_mais_perto_de_meio():
    model = FakeModel([0.05, 0.48, 0.95, 0.55, 0.2])
    chosen = uncertainty_sampling(model, pd.DataFrame(index=range(5)), n_queries=2)
    assert sorted(chosen) == [1, 3]


def test_cada_rodada_adiciona_os_rotulos_consultados():
    X, y = make_training_data()

    history = run_active_learning(
        X.iloc[:500], y.iloc[:500], X.iloc[500:], y.iloc[500:],
        strategy="incerteza", initial_fraction=0.2, n_queries=50, n_rounds=3,
    )

    assert list(history["rodada"]) == [0, 1, 2, 3]
    assert history["n_rotulados"].diff().dropna().eq(50).all()
