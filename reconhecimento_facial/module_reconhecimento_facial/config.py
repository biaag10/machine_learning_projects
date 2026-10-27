# CONFIGURAÇÕES DO SISTEMA DE RECONHECIMENTO FACIAL
#
# Todos os caminhos partem da raiz do projeto, então o sistema funciona
# em qualquer máquina, sem caminhos fixos.

from pathlib import Path

from loguru import logger

# =============================================
# CAMINHOS
# =============================================

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"

# Uma ou mais fotos de cada pessoa autorizada.
# O nome da pessoa vem do arquivo: "bianca.jpg", "bianca_2.jpg" -> "bianca".
# Também é possível usar uma subpasta por pessoa: autorizados/bianca/foto1.jpg
AUTHORIZED_DIR = DATA_DIR / "autorizados"

# Fotos que serão verificadas pelo sistema
TEST_DIR = DATA_DIR / "teste"

# Imagens com as bounding boxes desenhadas e relatório CSV
OUTPUT_DIR = PROJ_ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================
# MODELO (DeepFace)
# =============================================

# Modelo que transforma cada rosto em um vetor de 512 números (embedding)
MODEL_NAME = "Facenet512"

# Detector de rostos: RetinaFace (rede neural, mais precisa que o Haar Cascade do OpenCV:
# não confunde outras partes da imagem com rostos e lida melhor com rostos inclinados)
DETECTOR_BACKEND = "retinaface"

# Distância usada para comparar dois embeddings
DISTANCE_METRIC = "cosine"

# Rostos com confiança de detecção abaixo disso são descartados
MIN_FACE_CONFIDENCE = 0.5

# =============================================
# VISUALIZAÇÃO (cores em BGR, padrão do OpenCV)
# =============================================

GREEN = (0, 200, 0)
RED = (0, 0, 255)

LABEL_GRANTED = "ACESSO LIBERADO"
LABEL_DENIED = "ACESSO NEGADO"
