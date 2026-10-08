from loguru import logger
import numpy as np
from tensorflow import keras

from module_deep_learning.config import RAW_DATA_DIR


def download_dataset():

    logger.info(
        "Baixando a base CIFAR-10..."
    )

    (X_train, y_train), (X_test, y_test) = (
        keras.datasets.cifar10.load_data()
    )

    # y vem como coluna (N, 1) -> achata para (N,)
    return (
        X_train,
        y_train.flatten(),
        X_test,
        y_test.flatten(),
    )


def save_dataset(X_train, y_train, X_test, y_test, dataset_path):

    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    logger.success(
        f"Base salva em: {dataset_path}"
    )


def load_dataset(dataset_path=RAW_DATA_DIR / "cifar10.npz"):

    # Baixa e salva a base apenas na primeira execução
    if not dataset_path.exists():
        save_dataset(
            *download_dataset(),
            dataset_path=dataset_path,
        )

    with np.load(dataset_path) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    logger.info(
        f"Formato de X_train: {X_train.shape} | "
        f"Formato de X_test: {X_test.shape}"
    )

    return X_train, y_train, X_test, y_test


def main():
    load_dataset()


if __name__ == "__main__":
    main()