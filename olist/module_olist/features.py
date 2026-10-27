# BASE ANALÍTICA (1 linha por pedido, já com is_late)
#       ↓
# create_features()
#       ↓
# features de PRAZO       → promised_days, is_short_deadline
# features de TEMPO       → purchase_month, purchase_weekday, purchase_hour,
#                           is_weekend_purchase, approval_hours
# features de VALOR       → avg_item_price, freight_ratio
#       ↓
# BASE REFINADA (pronta para o split e para o pipeline)
#
# REGRA DE OURO (DATA LEAKAGE):
# a previsão acontece em order_approved_at. Só podemos usar o que já é
# conhecido nesse instante: datas de compra e aprovação, prazo prometido,
# itens, vendedores, preço, frete e estado do cliente.
# NUNCA usar: data de envio, data real de entrega, avaliação ou status final.

from loguru import logger
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Lista oficial de features do modelo (fonte única usada por split, pipeline
# e inferência, para que treino e previsão usem exatamente as mesmas colunas)
# -----------------------------------------------------------------------------

# Numéricas: entram sem transformação (modelos de árvore não precisam de scaling)
NUMERIC_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "approval_hours",
    "avg_item_price",
    "freight_ratio",
    "is_short_deadline",
    "is_weekend_purchase",
]

# Categóricas: recebem One-Hot Encoding no pipeline
CATEGORICAL_FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "customer_state",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "is_late"

# Limite de prazo curto, definido na EDA (pergunta 2):
# pedidos com até 7 dias de prazo atrasam ~28%, contra ~8% no geral.
SHORT_DEADLINE_DAYS = 7


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # =========================================================================
    # FEATURES ORIGINAIS (construídas na EDA)
    # =========================================================================

    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        data["order_estimated_delivery_date"]  # Data prometida para a entrega.
        - data["order_approved_at"]            # Data de aprovação do pagamento.
    ).dt.total_seconds().div(86_400)          # Converte segundos para dias.

    # Extrai o número do mês em que a compra foi realizada.
    # Exemplo: janeiro = 1, fevereiro = 2, ..., dezembro = 12.
    data["purchase_month"] = data["order_purchase_timestamp"].dt.month

    # Extrai o dia da semana em que a compra foi realizada.
    # 0 = segunda-feira ... 6 = domingo
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.dayofweek

    # Extrai a hora em que a compra foi realizada (0 a 23).
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour

    # =========================================================================
    # TRATAMENTO DE INCONSISTÊNCIA (notebook de qualidade dos dados)
    # =========================================================================

    # Prazo prometido menor ou igual a zero significa que a data prometida
    # ocorreu antes (ou no momento) da aprovação: registro inconsistente.
    invalid_deadline = data["promised_days"].le(0)
    logger.info(f"Pedidos removidos por prazo não positivo: {invalid_deadline.sum()}")
    data = data.loc[~invalid_deadline].copy()

    # =========================================================================
    # NOVAS FEATURES (atributos derivados)
    # =========================================================================

    # 1) is_short_deadline: prazo curto (até 7 dias).
    #    Justificativa: na EDA, a taxa de atraso salta para ~28% nessa faixa.
    #    Uma flag explícita ajuda o modelo a capturar esse "degrau" não linear.
    data["is_short_deadline"] = (
        data["promised_days"].le(SHORT_DEADLINE_DAYS).astype("int8")
    )

    # 2) approval_hours: horas entre a compra e a aprovação do pagamento.
    #    Justificativa: aprovações demoradas (ex.: boleto) consomem parte da
    #    folga logística. A taxa de atraso sobe de ~6,9% (aprovação em minutos)
    #    para ~9,7% (aprovação em mais de 21h).
    #    Sem leakage: a aprovação é exatamente o instante da previsão.
    data["approval_hours"] = (
        data["order_approved_at"] - data["order_purchase_timestamp"]
    ).dt.total_seconds().div(3_600)

    # 3) avg_item_price: preço médio por item.
    #    Justificativa: separa "pedido caro porque tem muitos itens" de
    #    "pedido com item caro" (produtos maiores/mais complexos). A taxa de
    #    atraso cresce de ~7,3% para ~9,4% do menor para o maior quintil.
    data["avg_item_price"] = data["total_price"] / data["item_count"]

    # 4) freight_ratio: peso do frete em relação ao valor dos produtos.
    #    Justificativa: frete proporcionalmente alto indica entrega distante ou
    #    produto volumoso. Sozinha é fraca, mas pode interagir com estado e prazo.
    #    np.where evita divisão por zero.
    data["freight_ratio"] = np.where(
        data["total_price"] > 0,
        data["total_freight"] / data["total_price"],
        0.0,
    )

    # 5) is_weekend_purchase: compra feita no sábado ou domingo.
    #    Justificativa: compras de fim de semana só começam a ser processadas
    #    no próximo dia útil, alterando a folga real do prazo.
    data["is_weekend_purchase"] = data["purchase_weekday"].ge(5).astype("int8")

    logger.info(f"Features criadas. Base refinada com {len(data):,} pedidos.")

    return data