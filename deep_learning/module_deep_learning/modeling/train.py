import json

from loguru import logger
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from module_deep_learning.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MIN_LR,
    MODELS_DIR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    REPORTS_DIR,
    SEED,
)
from module_deep_learning.dataset import load_dataset
from module_deep_learning.features import create_features
from module_deep_learning.modeling.model import build_model
from module_deep_learning.split import split_data


def get_callbacks(model_path):

    model_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        # Reduz a taxa de aprendizado quando a val_loss estaciona
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        # Para o treino e volta para os melhores pesos
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Salva o melhor modelo em disco
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_history(history, history_path):

    history_path.parent.mkdir(parents=True, exist_ok=True)

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    logger.info(
        f"Histórico salvo em: {history_path}"
    )


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_path=MODELS_DIR / "melhor_modelo_cifar10.keras",
    history_path=REPORTS_DIR / "history.json",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
):

    logger.info("Iniciando treinamento...")

    keras.utils.set_random_seed(SEED)

    model = build_model()

    result = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(model_path),
    )

    history = {
        key: [float(v) for v in values]
        for key, values in result.history.items()
    }

    save_history(history, history_path)

    logger.success(
        f"Treino concluído. Melhor val_accuracy: "
        f"{max(history['val_accuracy']) * 100:.2f}%"
    )

    logger.info(
        f"Modelo salvo em: {model_path}"
    )

    return model, history


def main():

    X_train, y_train, _, _ = load_dataset()

    X_train = create_features(X_train)

    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    train_model(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()