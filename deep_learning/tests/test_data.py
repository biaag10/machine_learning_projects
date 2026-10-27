import numpy as np

from module_deep_learning.config import IMG_SHAPE, NUM_CLASSES
from module_deep_learning.features import create_features
from module_deep_learning.modeling.model import build_model
from module_deep_learning.split import split_data


def test_pixels_sao_normalizados_entre_0_e_1():
    X = np.array([[[[0, 128, 255]]]], dtype="uint8")

    X_norm = create_features(X)

    assert X_norm.dtype == np.float32
    assert X_norm.min() == 0.0
    assert X_norm.max() == 1.0


def test_split_separa_validacao_sem_perder_imagens():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)

    X_train, X_val, y_train, y_val = split_data(X, y, val_size=0.2, seed=0)

    assert len(X_val) == 20
    assert len(X_train) == 80
    # Nenhuma imagem aparece nos dois conjuntos
    assert not set(y_train) & set(y_val)
    # Cada imagem continua com o seu rótulo
    assert (X_train.ravel() == y_train).all()


def test_modelo_gera_uma_probabilidade_por_classe():
    model = build_model()

    images = np.random.rand(2, *IMG_SHAPE).astype("float32")
    proba = model.predict(images, verbose=0)

    assert proba.shape == (2, NUM_CLASSES)
    # Softmax: as probabilidades de cada imagem somam 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
