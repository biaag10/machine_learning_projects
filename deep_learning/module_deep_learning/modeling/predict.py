from loguru import logger
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img

from module_deep_learning.config import (
    CLASS_NAMES,
    IMG_SHAPE,
    MODELS_DIR,
)
from module_deep_learning.features import create_features


def load_model(model_path=MODELS_DIR / "melhor_modelo_cifar10.keras"):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            f"Rode o treinamento antes."
        )

    logger.info(
        f"Carregando modelo de: {model_path}"
    )

    return keras.models.load_model(model_path)


def predict(model, X):

    # X precisa estar normalizado (create_features)
    probs = model.predict(X, verbose=0)

    y_pred = np.argmax(probs, axis=1)

    return probs, y_pred


def predict_image(model, image_path):

    # Classifica uma imagem qualquer do disco
    img = load_img(image_path, target_size=IMG_SHAPE[:2])

    X = create_features(
        img_to_array(img)[np.newaxis, ...]
    )

    probs, y_pred = predict(model, X)

    classe = CLASS_NAMES[y_pred[0]]
    confianca = float(probs[0, y_pred[0]])

    logger.info(
        f"{image_path.name}: {classe} ({confianca:.1%})"
    )

    return classe, confianca


def main():

    model = load_model()

    model.summary(print_fn=logger.info)


if __name__ == "__main__":
    main()