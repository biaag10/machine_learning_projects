from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import shap

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import (
    load_model,
)


def generate_explanations(data=None, model=None, model_name=None, sample_size=100):

    logger.info("Iniciando explicabilidade do modelo...")

    if data is None:
        data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    # Remove target das entradas
    X = data.drop(columns=["is_late"])

    # Seleciona algumas amostras
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    # Carrega o modelo treinado
    if model is None:
        model, model_name, _ = load_model(
            model_path=MODELS_DIR / "best_model.joblib",
            metadata_path=MODELS_DIR / "metadata.json",
        )

    logger.info(
        f"Modelo: {model_name}"
    )

    # Aplica o pré-processamento
    X_transformed = (
        prepare_data_for_shap(
            pipeline=model,
            X=X_sample,
        )
    )

    # Cria o explicador
    explainer = create_explainer(
        pipeline=model
    )

    # Calcula SHAP
    shap_values = (
        calculate_shap_values(
            explainer=explainer,
            X_transformed=X_transformed,
        )
    )

    logger.success(
        "Valores SHAP calculados."
    )
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.bar(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
         FIGURES_DIR /
        "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    sample_position = 0
    
    shap.plots.waterfall(
        shap_values[
            sample_position
        ],
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.success(
        "Gráficos SHAP gerados com sucesso."
    )

    logger.info(
        f"Gráficos salvos em: "
        f"{FIGURES_DIR}"
    )

    return [
        FIGURES_DIR / "shap_global_bar.png",
        FIGURES_DIR / "shap_beeswarm.png",
        FIGURES_DIR / "shap_waterfall.png",
    ]


def main():
    generate_explanations()

if __name__ == "__main__":
    main()
