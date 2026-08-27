# main.py
#
# Orquestrador do projeto Olist:
# 1 - Carrega os datasets brutos
# 2 - Cria dataset analítico
# 3 - Aplica feature engineering
# 4 - Salva dataset intermediário
# 5 - Divide treino/teste
# 6 - Treina modelos
# 7 - Avalia modelos


from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split

from module_olist.dataset import (
    load_dataset,
    create_dataset,
    save_dataset,
)

from module_olist.features import create_features

from module_olist.modeling.train import train_model
from module_olist.modeling.evaluate import evaluate_model



def main():

    # ============================================================
    # DEFINIÇÃO DOS CAMINHOS
    # ============================================================

    # Raiz do projeto Olist
    ROOT = Path(__file__).resolve().parents[1]

    # Diretórios
    DATA_RAW = ROOT / "data" / "raw"
    DATA_INTERIM = ROOT / "data" / "interim"


    DATA_INTERIM.mkdir(
        parents=True,
        exist_ok=True
    )


    # Arquivos de entrada

    orders_path = (
        DATA_RAW /
        "olist_orders_dataset.csv"
    )

    items_path = (
        DATA_RAW /
        "olist_order_items_dataset.csv"
    )

    customers_path = (
        DATA_RAW /
        "olist_customers_dataset.csv"
    )


    # Arquivo de saída

    output_path = (
        DATA_INTERIM /
        "olist_orders_features.csv"
    )


    logger.info("Iniciando pipeline Olist...")


    # ============================================================
    # CARREGAMENTO DOS DATASETS
    # ============================================================

    logger.info("Carregando datasets...")


    orders, items, customers = load_dataset(
        orders_path=orders_path,
        items_path=items_path,
        customers_path=customers_path,
    )


    logger.info(
        f"Orders carregado: {orders.shape}"
    )

    logger.info(
        f"Items carregado: {items.shape}"
    )

    logger.info(
        f"Customers carregado: {customers.shape}"
    )


    # ============================================================
    # CRIAÇÃO DATASET ANALÍTICO
    # ============================================================

    logger.info(
        "Criando dataset analítico..."
    )


    data = create_dataset(
        orders=orders,
        items=items,
        customers=customers,
    )


    logger.info(
        f"Dataset analítico criado: {data.shape}"
    )


    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    logger.info(
        "Criando features..."
    )


    data = create_features(data)


    logger.info(
        f"Dataset com features: {data.shape}"
    )


    # ============================================================
    # SALVAMENTO DATASET INTERMEDIÁRIO
    # ============================================================

    logger.info(
        "Salvando dataset intermediário..."
    )


    save_dataset(
        dataset=data,
        output_path=output_path,
    )


    logger.info(
        f"Dataset salvo em: {output_path}"
    )


    # ============================================================
    # PREPARAÇÃO PARA MODELAGEM
    # ============================================================

    logger.info(
        "Preparando dados para modelagem..."
    )


    # Remove variável alvo
    # e possíveis vazamentos de informação

    leakage_columns = [
        "is_late",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
    ]


    X = data.drop(
        columns=[
            col
            for col in leakage_columns
            if col in data.columns
        ]
    )


    y = data["is_late"]


    logger.info(
        f"Features utilizadas: {X.shape[1]}"
    )


    # ============================================================
    # TRAIN TEST SPLIT
    # ============================================================

    logger.info(
        "Separando treino e teste..."
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


    logger.info(
        f"Treino: {X_train.shape}"
    )

    logger.info(
        f"Teste: {X_test.shape}"
    )


    # ============================================================
    # TREINAMENTO DOS MODELOS
    # ============================================================

    logger.info(
        "Treinando modelos..."
    )


    models = train_model(
        X_train,
        y_train,
    )


    logger.success(
        "Treinamento finalizado."
    )


    # ============================================================
    # AVALIAÇÃO DOS MODELOS
    # ============================================================

    logger.info(
        "Avaliando modelos..."
    )


    results = evaluate_model(
        models,
        X_test,
        y_test,
    )


    logger.success(
        "Avaliação finalizada."
    )


    # Mostra resultados finais

    logger.info(
        f"Resultados: {results}"
    )


    logger.success(
        "Pipeline Olist finalizado com sucesso!"
    )



if __name__ == "__main__":
    main()