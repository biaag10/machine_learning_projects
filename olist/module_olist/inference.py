# INFERÊNCIA: prever o risco de atraso de pedidos NOVOS
#
# CSVs brutos (orders, items, customers)
#       ↓
# create_inference_dataset()  → só pedidos aprovados e ainda não entregues
#       ↓
# create_features()           → as MESMAS features do treino
#       ↓
# modelo salvo + threshold    → models/best_model.joblib + metadata.json
#       ↓
# probabilidade, previsão e faixa de risco por pedido
#       ↓
# models/open_orders_predictions.csv (ranking para a operação)

from loguru import logger
import pandas as pd

from module_olist.config import (
    MODELS_DIR,
    RAW_DATA_DIR,
)
from module_olist.dataset import create_inference_dataset, load_dataset
from module_olist.features import FEATURES, create_features
from module_olist.modeling.predict import (
    load_model,
    predict,
    validate_features,
)

# Colunas de identificação que acompanham a previsão no arquivo final
ID_COLUMNS = ["order_id", "order_status", "order_approved_at", "customer_state"]


def load_open_orders():
    """Carrega os dados brutos e monta a base de pedidos em aberto com as features."""

    orders, items, customers = load_dataset(
        orders_path=RAW_DATA_DIR / "olist_orders_dataset.csv",
        items_path=RAW_DATA_DIR / "olist_order_items_dataset.csv",
        customers_path=RAW_DATA_DIR / "olist_customers_dataset.csv",
    )

    data = create_inference_dataset(orders, items, customers)

    return create_features(data)


def run_inference(
    data=None,
    model=None,
    model_name=None,
    threshold=None,
    output_path=None,
):
    """
    Prevê o risco de atraso para pedidos ainda não entregues.

    - data: base com as features; se None, monta a base de pedidos em aberto.
    - model/threshold: se None, carrega o modelo salvo e o threshold da validação.
    """

    if data is None:
        data = load_open_orders()

    if model is None:
        model, model_name, threshold = load_model(
            model_path=MODELS_DIR / "best_model.joblib",
            metadata_path=MODELS_DIR / "metadata.json",
        )

    if threshold is None:
        raise ValueError("O threshold é obrigatório para realizar a inferência.")

    # Mesmas colunas, na mesma ordem, usadas no treino
    X = validate_features(data, FEATURES)

    predictions = predict(model=model, X=X, threshold=threshold)

    id_columns = [column for column in ID_COLUMNS if column in data.columns]

    result = (
        data[id_columns]
        .join(X[["promised_days", "total_price", "total_freight"]])
        .join(predictions)
        .sort_values("prob_is_late", ascending=False)
    )

    if output_path is None:
        output_path = MODELS_DIR / "open_orders_predictions.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    # Resumo para a operação
    logger.info(f"Modelo usado na inferência: {model_name} (threshold {threshold:.2f})")
    logger.info(f"Pedidos avaliados: {len(result):,}")
    logger.info(
        "Pedidos por faixa de risco:\n"
        f"{result['risk_level'].value_counts().reindex(['alto', 'médio', 'baixo'], fill_value=0).to_string()}"
    )
    logger.info(
        "10 pedidos com maior risco de atraso:\n"
        f"{result.head(10)[['order_id', 'customer_state', 'promised_days', 'prob_is_late', 'risk_level']].to_string(index=False)}"
    )
    logger.success(f"Previsões salvas em: {output_path}")

    return result


def main():
    run_inference()


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()