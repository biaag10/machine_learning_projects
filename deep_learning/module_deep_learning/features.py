from loguru import logger
from tensorflow import keras
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomTranslation,
)

from module_deep_learning.dataset import load_dataset


def create_features(X):

    # Normaliza os pixels de [0, 255] para [0, 1]
    return X.astype("float32") / 255.0


def build_data_augmentation():

    # Só atua no treino (fit); em evaluate/predict
    # as imagens passam sem alteração
    return keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomTranslation(0.1, 0.1),
            RandomRotation(0.05),
        ],
        name="data_augmentation",
    )


def main():

    X_train, _, _, _ = load_dataset()

    X_train = create_features(X_train)

    logger.info(
        f"Pixels normalizados: "
        f"min={X_train.min():.2f} | "
        f"max={X_train.max():.2f}"
    )


if __name__ == "__main__":
    main()