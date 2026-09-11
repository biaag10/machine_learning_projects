from loguru import logger
import pandas as pd

from module_olist.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.predict import (
    load_model,
    predict,
)

FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
]


def run_inference(
    data=None,
    model=None,
    model_name=None,
    threshold=None,
    sample_size=5,
    output_path=None,
):
    """Executa uma inferência de exemplo e salva as previsões."""

    if data is None:
        data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    X = data[FEATURES]
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    if model is None:
        model, model_name, threshold = load_model(
            model_path=MODELS_DIR / "best_model.joblib",
            metadata_path=MODELS_DIR / "metadata.json",
        )

    if threshold is None:
        raise ValueError("O threshold é obrigatório para realizar a inferência.")

    predictions = predict(model=model, X=X_sample, threshold=threshold)
    result = X_sample.join(predictions)

    if output_path is None:
        output_path = MODELS_DIR / "sample_predictions.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=True)

    logger.info(f"Modelo usado na inferência: {model_name}")
    logger.info(f"Amostras selecionadas para inferência:\n{X_sample}")
    logger.success(f"Predições realizadas:\n{predictions}")
    logger.success(f"Predições salvas em: {output_path}")

    return result


def main():
    run_inference()


if __name__ == "__main__":
    main()
