# LEITURA E GRAVAÇÃO DE IMAGENS COM SUPORTE A ACENTOS
#
# No Windows, cv2.imread e cv2.imwrite não abrem caminhos com acentos
# (ex.: "D:\OneDrive\Área de Trabalho\..."), e o DeepFace recusa esses caminhos.
# Por isso lemos os bytes do arquivo com numpy e decodificamos com o OpenCV:
# o DeepFace passa a receber a imagem já carregada (array BGR), não o caminho.

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path):
    """Lê uma imagem (BGR) de qualquer caminho. Retorna None se não for uma imagem válida."""
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    """Salva uma imagem (BGR) em qualquer caminho, inclusive com acentos."""
    ok, buffer = cv2.imencode(Path(path).suffix or ".jpg", image)

    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem para {path}")

    buffer.tofile(str(path))
