from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carrega variáveis de ambiente do arquivo .env, se existir
load_dotenv()

# =============================================
# CAMINHOS
# =============================================

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================
# DADOS
# =============================================

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]

# =============================================
# HIPERPARÂMETROS
# =============================================

SEED = 42
VAL_SIZE = 0.1

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
L2_WEIGHT = 1e-4

# Callbacks
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-5
EARLY_STOPPING_PATIENCE = 12

# Faz o loguru conviver com as barras de progresso do tqdm
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass