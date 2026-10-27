# RECONHECIMENTO
#
# imagem de teste
#       ↓
# DeepFace.represent()  → um embedding para CADA rosto encontrado
#       ↓
# distância de cosseno até cada pessoa autorizada
#       ↓
# menor distância <= threshold ?  →  ACESSO LIBERADO (nome)
#                                 →  ACESSO NEGADO

from deepface import DeepFace
from deepface.modules.verification import find_cosine_distance, find_threshold
import numpy as np

from module_reconhecimento_facial.config import (
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MIN_FACE_CONFIDENCE,
    MODEL_NAME,
)

# Threshold calibrado pelo DeepFace para o par modelo + métrica.
# Distâncias abaixo dele são consideradas a mesma pessoa.
THRESHOLD = find_threshold(MODEL_NAME, DISTANCE_METRIC)


def match_face(embedding, database, threshold=THRESHOLD) -> dict:
    """
    Compara um rosto com todas as fotos autorizadas.

    Retorna a pessoa mais parecida, a distância e se o acesso foi liberado.
    """
    embedding = np.asarray(embedding)

    distances = [
        float(find_cosine_distance(entry["embedding"], embedding)) for entry in database
    ]
    best = int(np.argmin(distances))

    return {
        "granted": distances[best] <= threshold,
        "name": database[best]["name"],
        "distance": distances[best],
        "threshold": threshold,
    }


def detect_faces(image):
    """
    Detecta os rostos da imagem (BGR) e gera o embedding de cada um.

    enforce_detection=False: se não houver rosto, o DeepFace devolve a imagem
    inteira com confiança 0; esses casos são descartados aqui.
    """
    faces = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
    )

    return [face for face in faces if face.get("face_confidence", 0) >= MIN_FACE_CONFIDENCE]


def recognize(image, database) -> list[dict]:
    """
    Reconhece todos os rostos de uma imagem.

    Cada resultado tem a posição do rosto (x, y, w, h) e a decisão de acesso.
    """
    results = []

    for face in detect_faces(image):
        area = face["facial_area"]
        decision = match_face(face["embedding"], database)

        results.append(
            {
                "x": area["x"],
                "y": area["y"],
                "w": area["w"],
                "h": area["h"],
                **decision,
            }
        )

    return results
