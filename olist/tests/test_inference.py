import numpy as np
import pandas as pd
import pytest

from module_olist.dataset import create_inference_dataset
from module_olist.modeling.predict import classify_risk, validate_features


def test_faixas_de_risco_usam_o_threshold():
    risk = classify_risk(np.array([0.05, 0.15, 0.30]), threshold=0.14)
    assert list(risk) == ["baixo", "médio", "alto"]


def test_inferencia_exige_as_colunas_do_treino():
    X = pd.DataFrame({"promised_days": [10.0]})
    with pytest.raises(ValueError, match="customer_state"):
        validate_features(X, ["promised_days", "customer_state"])


def test_base_de_inferencia_tem_so_pedidos_em_aberto_e_aprovados():
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "order_status": ["shipped", "delivered", "processing", "created"],
            "order_purchase_timestamp": pd.to_datetime(["2018-08-01"] * 4),
            "order_approved_at": pd.to_datetime(
                ["2018-08-01", "2018-08-01", "2018-08-02", None]
            ),
            "order_estimated_delivery_date": pd.to_datetime(["2018-08-20"] * 4),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            "order_item_id": [1, 1, 1, 1],
            "seller_id": ["s1"] * 4,
            "price": [100.0] * 4,
            "freight_value": [10.0] * 4,
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "customer_state": ["BA", "SP", "RJ", "MG"],
            "customer_city": ["x"] * 4,
        }
    )

    data = create_inference_dataset(orders, items, customers)

    # "delivered" já tem resultado e "created" ainda não foi aprovado
    assert sorted(data["order_id"]) == ["a", "c"]
    assert "is_late" not in data.columns
