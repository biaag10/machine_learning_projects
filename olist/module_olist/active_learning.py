# module_olist/active_learning.py
#
# ACTIVE LEARNING (pool-based, com human-in-the-loop simulado)
#
# Cenário da aula: temos ~100 mil pedidos, mas só sabemos se ~10% atrasaram.
# Rotular custa caro (alguém precisa acompanhar a entrega). Então perguntamos:
#
#   "Modelo, quais pedidos você NÃO consegue decidir e gostaria que um
#    humano rotulasse?"
#
# Fluxo:
#
#   treino (pool) ──► 10% ROTULADO  +  90% SEM RÓTULO
#        │
#        ▼
#   treina modelo ─► avalia no teste ─► mede a incerteza no pool sem rótulo
#        ▲                                        │
#        └── humano (oráculo) rotula os mais incertos ◄┘
#
# Para provar que vale a pena, comparamos duas estratégias de consulta:
#   - "incerteza": escolhe os pedidos com probabilidade mais perto de 0.5
#                  (menor Maximum Predicted Probability, como no slide)
#   - "aleatoria": escolhe pedidos ao acaso (baseline)
#
# Se o active learning funciona, a curva da incerteza sobe mais rápido:
# o modelo chega na mesma métrica pedindo MENOS rótulos ao humano.
#
# Uso (depois de rodar o main, que gera a base refinada):
#     python -m module_olist.active_learning

from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from module_olist.pipeline import create_lightgbm_pipeline
from module_olist.split import split_data


# ---------------------------------------------------------------------------
# Medida de confiança e estratégias de seleção
# ---------------------------------------------------------------------------


def max_predicted_probability(model, X_pool: pd.DataFrame) -> np.ndarray:
    """
    Maximum Predicted Probability (MPP): a maior probabilidade entre as classes.
    Perto de 1.0 = modelo confiante. Perto de 0.5 = modelo em dúvida.
    """
    proba = model.predict_proba(X_pool)
    return proba.max(axis=1)


def uncertainty_sampling(model, X_pool: pd.DataFrame, n_queries: int, rng=None) -> np.ndarray:
    """Seleciona os n_queries exemplos com MENOR confiança (Top-k mais incertos)."""
    confidence = max_predicted_probability(model, X_pool)
    return np.argsort(confidence)[:n_queries]


def random_sampling(model, X_pool: pd.DataFrame, n_queries: int, rng=None) -> np.ndarray:
    """Baseline: escolhe exemplos do pool ao acaso."""
    rng = rng if rng is not None else np.random.default_rng(42)
    return rng.choice(len(X_pool), size=min(n_queries, len(X_pool)), replace=False)


STRATEGIES = {
    "incerteza": uncertainty_sampling,
    "aleatoria": random_sampling,
}


# ---------------------------------------------------------------------------
# Loop de active learning
# ---------------------------------------------------------------------------


def run_active_learning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "incerteza",
    initial_fraction: float = 0.10,
    n_queries: int = 500,
    n_rounds: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simula o active learning e devolve, por rodada, quantos rótulos foram
    usados e as métricas no conjunto de teste.

    - initial_fraction: fração do treino que já começa rotulada (10%, como na aula)
    - n_queries: quantos pedidos o "humano" rotula por rodada
    - n_rounds: quantas rodadas de consulta
    """
    rng = np.random.default_rng(random_state)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Conjunto inicial rotulado, estratificado para ter as duas classes
    n_initial = int(len(X_train) * initial_fraction)
    labeled = (
        y_train.groupby(y_train, group_keys=False)
        .apply(lambda s: s.sample(frac=n_initial / len(y_train), random_state=random_state))
        .index.to_numpy()
    )
    unlabeled = np.setdiff1d(np.arange(len(X_train)), labeled)

    query_fn = STRATEGIES[strategy]
    history = []

    for round_ in range(n_rounds + 1):
        # 1. Treina só com o que já tem rótulo
        model = create_lightgbm_pipeline()
        model.set_params(model__verbose=-1)
        model.fit(X_train.iloc[labeled], y_train.iloc[labeled])

        # 2. Avalia no teste (PR-AUC é a métrica principal: classes desbalanceadas)
        proba_test = model.predict_proba(X_test)[:, 1]
        history.append(
            {
                "estrategia": strategy,
                "rodada": round_,
                "n_rotulados": len(labeled),
                "roc_auc": roc_auc_score(y_test, proba_test),
                "pr_auc": average_precision_score(y_test, proba_test),
            }
        )
        logger.info(
            f"[{strategy}] rodada {round_}: {len(labeled)} rótulos | "
            f"ROC-AUC={history[-1]['roc_auc']:.3f} | PR-AUC={history[-1]['pr_auc']:.3f}"
        )

        if round_ == n_rounds or len(unlabeled) == 0:
            break

        # 3. Modelo escolhe quais pedidos quer que o humano rotule
        chosen = query_fn(model, X_train.iloc[unlabeled], n_queries, rng)
        new_idx = unlabeled[chosen]

        # 4. Human-in-the-loop: o "oráculo" revela y (aqui simulado com o y real)
        labeled = np.concatenate([labeled, new_idx])
        unlabeled = np.setdiff1d(unlabeled, new_idx)

    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Comparação e relatório
# ---------------------------------------------------------------------------


def full_label_reference(X_train, y_train, X_test, y_test) -> float:
    """
    Referência: PR-AUC do mesmo modelo treinado com TODOS os rótulos.
    Mostra o quanto o active learning chega perto do cenário ideal
    usando só uma fração dos rótulos.
    """
    model = create_lightgbm_pipeline()
    model.set_params(model__verbose=-1)
    model.fit(X_train, y_train)
    return average_precision_score(y_test, model.predict_proba(X_test)[:, 1])


def plot_learning_curves(
    results: pd.DataFrame,
    output_path: Path,
    reference_pr_auc: float | None = None,
    n_total: int | None = None,
) -> None:
    """Plota PR-AUC x número de pedidos rotulados para cada estratégia."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, df in results.groupby("estrategia"):
        ax.plot(df["n_rotulados"], df["pr_auc"], marker="o", label=strategy)

    if reference_pr_auc is not None:
        label = "todos os rótulos"
        if n_total is not None:
            label += f" ({n_total:,} pedidos)".replace(",", ".")
        ax.axhline(reference_pr_auc, color="gray", linestyle="--", label=label)

    ax.set_xlabel("Pedidos rotulados")
    ax.set_ylabel("PR-AUC no teste")
    ax.set_title("Active learning: incerteza x amostragem aleatória")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Curva salva em: {output_path}")


def main():
    logger.info("Iniciando simulação de active learning...")

    data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")
    X_train, X_test, y_train, y_test = split_data(data)

    results = pd.concat(
        [
            run_active_learning(X_train, y_train, X_test, y_test, strategy=s)
            for s in STRATEGIES
        ],
        ignore_index=True,
    )

    reference = full_label_reference(X_train, y_train, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "active_learning_results.csv", index=False)
    plot_learning_curves(
        results,
        FIGURES_DIR / "active_learning_curve.png",
        reference_pr_auc=reference,
        n_total=len(X_train),
    )

    final = results.groupby("estrategia").tail(1)
    logger.success(f"Resultado final:\n{final.to_string(index=False)}")

    used = final["n_rotulados"].iloc[0] / len(X_train)
    logger.info(
        f"Referência com todos os {len(X_train):,} rótulos: PR-AUC={reference:.3f}. "
        f"O active learning usou {used:.0%} dos rótulos."
    )


if __name__ == "__main__":
    main()
