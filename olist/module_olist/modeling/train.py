import pandas as pd

from module_olist.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
):
    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' não encontrado. "
            f"Modelos disponíveis: {list(models.keys())}"
        )

    model = models[model_name]

    model.fit(
        X_train,
        y_train,
    )

    return model