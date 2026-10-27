import pandas as pd

from module_olist.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    create_features,
)

# Colunas que só existem DEPOIS do instante da previsão (aprovação).
# Se alguma delas virar feature, o modelo "cola" do futuro (data leakage).
LEAKAGE_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "is_late",
]


def make_orders():
    """Três pedidos de exemplo; o terceiro tem prazo inconsistente (negativo)."""
    return pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-03-03 10:00", "2018-03-05 08:00", "2018-03-06 09:00"]
            ),
            "order_approved_at": pd.to_datetime(
                ["2018-03-04 10:00", "2018-03-05 08:30", "2018-03-06 10:00"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-03-09", "2018-03-30", "2018-03-01"]
            ),
            "item_count": [2, 1, 1],
            "seller_count": [1, 1, 1],
            "total_price": [100.0, 50.0, 30.0],
            "total_freight": [20.0, 10.0, 5.0],
            "customer_state": ["BA", "SP", "RJ"],
        }
    )


def test_nenhuma_feature_usa_informacao_futura():
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS)


def test_listas_de_features_sao_consistentes():
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(FEATURES) == len(set(FEATURES))


def test_create_features_gera_todas_as_colunas():
    data = create_features(make_orders())
    assert set(FEATURES).issubset(data.columns)


def test_remove_prazo_nao_positivo():
    data = create_features(make_orders())
    assert len(data) == 2
    assert data["promised_days"].gt(0).all()


def test_valores_das_novas_features():
    data = create_features(make_orders()).reset_index(drop=True)

    # Pedido 1: comprado num sábado, aprovado 24h depois, prazo de 5 dias
    assert data.loc[0, "approval_hours"] == 24
    assert data.loc[0, "is_weekend_purchase"] == 1
    assert data.loc[0, "is_short_deadline"] == 1
    assert data.loc[0, "avg_item_price"] == 50
    assert data.loc[0, "freight_ratio"] == 0.2

    # Pedido 2: segunda-feira, prazo longo
    assert data.loc[1, "is_weekend_purchase"] == 0
    assert data.loc[1, "is_short_deadline"] == 0
