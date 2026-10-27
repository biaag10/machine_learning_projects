# BASE DE PESSOAS AUTORIZADAS
#
# data/autorizados/*.jpg
#       ↓
# DeepFace.represent()  → detecta o rosto (RetinaFace) e gera o embedding (Facenet512)
#       ↓
# lista de (nome, embedding) usada para comparar com as fotos de teste

from pathlib import Path

from deepface import DeepFace
from loguru import logger
import numpy as np

from module_reconhecimento_facial.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    IMAGE_EXTENSIONS,
    MODEL_NAME,
)
from module_reconhecimento_facial.image_io import read_image


def list_images(folder: Path) -> list[Path]:
    """Lista as imagens de uma pasta (inclusive subpastas), em ordem alfabética."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def person_name(image_path: Path, base_dir: Path = AUTHORIZED_DIR) -> str:
    """
    Descobre o nome da pessoa a partir do caminho da foto.

    - autorizados/bianca/foto1.jpg -> "bianca"  (subpasta = pessoa)
    - autorizados/bianca_2.jpg     -> "bianca"  (tudo antes do primeiro "_")
    """
    relative = image_path.relative_to(base_dir)

    if len(relative.parts) > 1:
        return relative.parts[0]

    return image_path.stem.split("_")[0]


def largest_face(faces: list[dict]) -> dict:
    """Em uma foto de cadastro, usa o maior rosto (a pessoa em primeiro plano)."""
    return max(faces, key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"])


def build_database(authorized_dir: Path = AUTHORIZED_DIR) -> list[dict]:
    """
    Gera o embedding de cada foto autorizada.

    Retorna uma lista de dicionários: {"name", "embedding", "source"}.
    """
    images = list_images(authorized_dir)

    if not images:
        raise FileNotFoundError(
            f"Nenhuma foto encontrada em {authorized_dir}. "
            "Adicione pelo menos uma foto de cada pessoa autorizada."
        )

    database = []

    for image_path in images:
        image = read_image(image_path)

        if image is None:
            logger.warning(f"Não foi possível abrir {image_path.name}: foto ignorada.")
            continue

        try:
            faces = DeepFace.represent(
                img_path=image,  # imagem já carregada: funciona com acentos no caminho
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,  # foto de cadastro PRECISA ter um rosto
            )
        except ValueError as error:
            logger.warning(f"{image_path.name}: foto ignorada ({error})")
            continue

        face = largest_face(faces)
        name = person_name(image_path, authorized_dir)

        database.append(
            {
                "name": name,
                "embedding": np.asarray(face["embedding"]),
                "source": image_path.name,
            }
        )
        logger.info(f"Autorizado cadastrado: {name} ({image_path.name})")

    if not database:
        raise ValueError("Nenhuma foto autorizada teve um rosto detectado.")

    people = sorted({entry["name"] for entry in database})
    logger.success(f"{len(database)} foto(s) de {len(people)} pessoa(s) autorizada(s): {people}")

    return database
