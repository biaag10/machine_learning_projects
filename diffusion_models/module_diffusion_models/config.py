# CONFIGURAÇÕES DO PROJETO DIFFUSION MODELS
#
# Todos os caminhos partem da raiz do projeto, e os hiperparâmetros são os mesmos
# do experimento apresentado nos slides (notebooks/Diffusion_Models_MNIST_SD_executado.ipynb).

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

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
MODEL_PATH = MODELS_DIR / "ddpm_mnist_ema.pt"  # pesos (EMA) do modelo treinado

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================
# DADOS
# =============================================

IMG_SIZE = 32  # MNIST 28x28 redimensionado para 32x32 (divisível por 2 nas descidas da U-Net)
NUM_CLASSES = 10  # dígitos de 0 a 9; a classe extra 10 significa "sem condição" (CFG)

# =============================================
# DIFUSÃO (Ho et al., 2020)
# =============================================

T = 1000  # nº de passos do processo direto
BETA_START = 1e-4  # variância do ruído no primeiro passo
BETA_END = 0.02  # variância do ruído no último passo

# =============================================
# MODELO E TREINO
# =============================================

CH = 32  # largura da U-Net: 32 -> ~0,75 M parâmetros (nosso experimento)
EPOCHS = 10
BATCH_SIZE = 128
LR = 2e-4
P_UNCOND = 0.1  # em 10% do treino a classe é escondida (Classifier-Free Guidance)
EMA_DECAY = 0.999  # média móvel dos pesos, usada para gerar as imagens
SEED = 42

# =============================================
# GERAÇÃO
# =============================================

GUIDANCE = 3.0  # w = 3 foi o melhor equilíbrio entre acurácia e FID
DDIM_STEPS = 50  # DDIM com 50 passos: ~19x mais rápido que o DDPM com 1000


def get_device() -> str:
    """Usa a GPU quando houver; senão, a CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
