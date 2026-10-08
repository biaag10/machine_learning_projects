from loguru import logger
import numpy as np

from module_deep_learning.config import SEED, VAL_SIZE
from module_deep_learning.dataset import load_dataset


def split_data(X, y, val_size=VAL_SIZE, seed=SEED):

    # Embaralha os índices e separa uma parte para validação
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_val = int(len(X) * val_size)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info(
        f"Treino: {len(X_train)} imagens | "
        f"Validação: {len(X_val)} imagens"
    )

    return X_train, X_val, y_train, y_val


def main():

    X_train, y_train, _, _ = load_dataset()

    split_data(X_train, y_train)


if __name__ == "__main__":
    main()