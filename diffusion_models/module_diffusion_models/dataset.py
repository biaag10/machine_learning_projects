# DADOS: MNIST
#
# 70 mil dígitos manuscritos 28x28 (60 mil de treino e 10 mil de teste).
# Redimensionamos para 32x32 e normalizamos para [-1, 1], a mesma escala do ruído gaussiano.
#
# Uso:  python -m module_diffusion_models.dataset   (baixa o MNIST em data/raw)

from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import typer

from module_diffusion_models.config import BATCH_SIZE, IMG_SIZE, RAW_DATA_DIR

app = typer.Typer()


def get_transform():
    """28x28 -> 32x32, tensor em [0, 1] -> [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_mnist(train: bool = True, data_dir: Path = RAW_DATA_DIR, download: bool = True):
    """Carrega o MNIST de treino (train=True) ou de teste (train=False)."""
    return datasets.MNIST(str(data_dir), train=train, download=download, transform=get_transform())


def get_dataloader(dataset, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """DataLoader de treino: embaralha e descarta o último lote incompleto."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@app.command()
def main(data_dir: Path = RAW_DATA_DIR):
    logger.info("Baixando o MNIST...")
    train_ds = load_mnist(train=True, data_dir=data_dir)
    test_ds = load_mnist(train=False, data_dir=data_dir)
    logger.success(f"MNIST pronto em {data_dir}: {len(train_ds)} treino | {len(test_ds)} teste")


if __name__ == "__main__":
    app()
