from module_olist.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olist.dataset import (
    load_data,
)

from module_olist.modeling.predict import (
    load_model,
    predict,
)

from loguru import logger
import pandas as pd


def main():

    # Carrega o dataset já preparado
    data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    # somente as features usadas no treinamento
    X = data[
        [
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
    ]

    # Seleciona algumas amostras
    X_sample = X.sample(n=5, random_state=42,)

    # Carrega o modelo já treinado
    model, model_name, threshold = (
        load_model(
            model_path=(
                MODELS_DIR /
                "best_model.joblib"
            ),
            metadata_path=(
                MODELS_DIR /
                "metadata.json"
            ),
        )
    )

    # Realiza a inferência
    predictions = predict(
        model=model,
        X=X_sample,
        threshold=threshold,
    )

    logger.info(f"Amostras selecionadas para inferência: {X_sample}\n")

    logger.success(f"Predições realizadas:\n{predictions}")


if __name__ == "__main__":
    main()