import pandas as pd
from module_olist.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def train_model(X_train, y_train):

    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

    return trained_models