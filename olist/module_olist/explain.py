# EXPLICABILIDADE (SHAP)
#
# A previsão diz QUAIS pedidos têm risco. A explicação diz POR QUÊ:
# foi o prazo prometido? o estado do cliente? o frete?
#
# GLOBAL (o modelo como um todo)
#   - shap_global_bar.png  → importância média de cada feature
#   - shap_beeswarm.png    → direção do efeito (valor alto/baixo aumenta o risco?)
#   - reports/shap_importance.csv
#
# LOCAL (um pedido específico)
#   - shap_waterfall.png   → por que o pedido EM ABERTO de maior risco foi sinalizado
#   - reports/open_orders_explanations.csv → 3 principais fatores de cada pedido de risco alto
#
# Leitura dos valores SHAP: o XGBoost trabalha em log-odds. Valor SHAP positivo
# empurra a previsão para "atraso"; negativo empurra para "no prazo".
#
# LIMITES: SHAP explica o que o MODELO aprendeu, não a causa real do atraso.
# Se o estado aparece como importante, ele pode estar representando distância
# ou infraestrutura logística (correlação, não causalidade).

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from module_olist.features import CATEGORICAL_FEATURES, FEATURES
from module_olist.modeling.interpret import (
    aggregate_shap_by_feature,
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import (
    load_model,
)

MAX_DISPLAY = 10


def explain_rows(model, X):
    """Calcula o SHAP já agregado por feature original para as linhas de X."""

    X_transformed = prepare_data_for_shap(pipeline=model, X=X)
    explainer = create_explainer(pipeline=model)
    shap_values = calculate_shap_values(explainer=explainer, X_transformed=X_transformed)

    return aggregate_shap_by_feature(shap_values, X, CATEGORICAL_FEATURES)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def top_factors(explanation, row, n=3):
    """
    Texto com os n fatores que mais AUMENTARAM o risco de um pedido.
    Ex.: "promised_days=2.3 (+0.85); customer_state=AL (+0.40)"
    """

    contributions = pd.Series(explanation.values[row], index=explanation.feature_names)
    values = pd.Series(explanation.display_data[row], index=explanation.feature_names)

    positives = contributions[contributions > 0].sort_values(ascending=False).head(n)

    def fmt(value):
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    return "; ".join(f"{name}={fmt(values[name])} (+{shap:.2f})" for name, shap in positives.items())


def explain_open_orders(model, risk_levels=("alto",)):
    """
    Explicação LOCAL dos pedidos em aberto sinalizados como risco alto,
    usando as previsões geradas pela inferência.
    Retorna a explicação do pedido de MAIOR risco (para o waterfall).
    """

    predictions_path = MODELS_DIR / "open_orders_predictions.csv"

    if not predictions_path.exists():
        logger.warning(
            "Previsões de pedidos em aberto não encontradas. "
            "Rode a inferência antes: python -m module_olist.inference"
        )
        return None

    # Reconstrói as features dos pedidos em aberto (as mesmas da inferência)
    from module_olist.inference import load_open_orders

    open_orders = load_open_orders()
    predictions = pd.read_csv(predictions_path)

    risky = predictions.loc[predictions["risk_level"].isin(risk_levels)]
    risky_orders = (
        open_orders.set_index("order_id")
        .loc[risky["order_id"]]
        .reset_index()
    )

    explanation = explain_rows(model, risky_orders[FEATURES])

    report = risky[["order_id", "customer_state", "prob_is_late", "risk_level"]].copy()
    report["principais_fatores"] = [top_factors(explanation, i) for i in range(len(report))]

    output_path = REPORTS_DIR / "open_orders_explanations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    logger.info(
        "Por que os pedidos de maior risco foram sinalizados:\n"
        f"{report.head(5)[['order_id', 'prob_is_late', 'principais_fatores']].to_string(index=False)}"
    )
    logger.success(f"Explicações dos pedidos de risco alto salvas em: {output_path}")

    # O primeiro é o de maior probabilidade (as previsões estão ordenadas)
    return explanation[0], report.iloc[0]["order_id"]


def generate_explanations(data=None, model=None, model_name=None, sample_size=1000):

    logger.info("Iniciando explicabilidade do modelo...")

    if data is None:
        data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    # Somente as features do modelo (sem alvo e sem datas futuras)
    X = data[FEATURES]

    # Amostra para as explicações globais
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    # Carrega o modelo treinado
    if model is None:
        model, model_name, _ = load_model(
            model_path=MODELS_DIR / "best_model.joblib",
            metadata_path=MODELS_DIR / "metadata.json",
        )

    logger.info(f"Modelo: {model_name}")

    # =============================================
    # EXPLICAÇÃO GLOBAL
    # =============================================

    explanation = explain_rows(model, X_sample)

    logger.success("Valores SHAP calculados (agregados por feature original).")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Importância global: média do |SHAP| de cada feature
    importance = (
        pd.DataFrame(abs(explanation.values), columns=explanation.feature_names)
        .mean()
        .sort_values(ascending=False)
        .to_frame("mean_abs_shap")
        .round(4)
    )
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index_label="feature")

    logger.info(f"Importância global (média |SHAP|):\n{importance.to_string()}")

    shap.plots.bar(explanation, max_display=MAX_DISPLAY, show=False)
    plt.title(f"Importância global das features - {model_name}")
    save_figure(FIGURES_DIR / "shap_global_bar.png")

    shap.plots.beeswarm(explanation, max_display=MAX_DISPLAY, show=False)
    plt.title("Efeito de cada feature no risco de atraso")
    save_figure(FIGURES_DIR / "shap_beeswarm.png")

    # =============================================
    # EXPLICAÇÃO LOCAL (pedido em aberto de maior risco)
    # =============================================

    local = explain_open_orders(model)

    if local is not None:
        order_explanation, order_id = local
        title = f"Por que o pedido {order_id[:8]}... tem alto risco de atraso"
    else:
        # Sem previsões de pedidos em aberto: explica um pedido da amostra
        order_explanation = explanation[0]
        title = "Explicação de um pedido da amostra"

    shap.plots.waterfall(order_explanation, max_display=MAX_DISPLAY, show=False)
    plt.title(title)
    save_figure(FIGURES_DIR / "shap_waterfall.png")

    logger.success("Gráficos SHAP gerados com sucesso.")
    logger.info(f"Gráficos salvos em: {FIGURES_DIR}")

    return [
        FIGURES_DIR / "shap_global_bar.png",
        FIGURES_DIR / "shap_beeswarm.png",
        FIGURES_DIR / "shap_waterfall.png",
    ]


def main():
    generate_explanations()


if __name__ == "__main__":
    main()
