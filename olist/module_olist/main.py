# main.py
#
# Orquestrador do projeto Olist:
# 1 - Carrega os datasets brutos
# 2 - Cria dataset analítico
# 3 - Aplica feature engineering
# 4 - Salva dataset intermediário
# 5 - Divide treino/teste
# 6 - Executa validação cruzada
# 7 - Seleciona o melhor modelo e threshold
# 8 - Treina o modelo final
# 9 - Avalia o modelo no conjunto de teste


from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split

from module_olist.dataset import (
    load_dataset,
    create_dataset,
    save_dataset,
)

from module_olist.features import create_features

from module_olist.modeling.cross_validation import (
    cross_validate_models,
)

from module_olist.modeling.train import (
    train_model,
)

from module_olist.modeling.evaluate import (
    evaluate_model,
)


def main():

    # ============================================================
    # DEFINIÇÃO DOS CAMINHOS
    # ============================================================

    # Raiz do projeto Olist
    ROOT = Path(__file__).resolve().parents[1]

    # Diretórios de dados
    DATA_RAW = ROOT / "data" / "raw"
    DATA_INTERIM = ROOT / "data" / "interim"

    # Cria a pasta interim caso não exista
    DATA_INTERIM.mkdir(
        parents=True,
        exist_ok=True,
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

    logger.info(
        "Iniciando pipeline Olist..."
    )


    # ============================================================
    # CARREGAMENTO DOS DATASETS
    # ============================================================

    logger.info(
        "Carregando datasets..."
    )

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
    # CRIAÇÃO DO DATASET ANALÍTICO
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
    # SALVAMENTO DO DATASET INTERMEDIÁRIO
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

    # Variável alvo e possíveis colunas
    # que representam informações posteriores
    # ao momento da previsão.

    leakage_columns = [
        "is_late",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
    ]

    # Features
    X = data.drop(
        columns=[
            column
            for column in leakage_columns
            if column in data.columns
        ]
    )

    # Target
    y = data["is_late"]

    logger.info(
        f"Quantidade de features utilizadas: {X.shape[1]}"
    )

    logger.info(
        f"Quantidade de observações: {X.shape[0]}"
    )


    # ============================================================
    # TRAIN / TEST SPLIT
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
    # CROSS VALIDATION
    # ============================================================

    logger.info(
        "Iniciando validação cruzada..."
    )

    best_model_name, best_threshold = (
        cross_validate_models(
            X_train=X_train,
            y_train=y_train,
        )
    )

    logger.success(
        f"Melhor modelo selecionado: "
        f"{best_model_name}"
    )

    logger.success(
        f"Melhor threshold: "
        f"{best_threshold:.2f}"
    )


    # ============================================================
    # TREINAMENTO FINAL
    # ============================================================

    logger.info(
        "Treinando modelo final..."
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=best_model_name,
    )

    logger.success(
        f"Modelo final treinado: "
        f"{best_model_name}"
    )


    # ============================================================
    # AVALIAÇÃO FINAL
    # ============================================================

    logger.info(
        "Avaliando modelo no conjunto de teste..."
    )

    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=best_threshold,
    )

    logger.success(
        "Avaliação final concluída."
    )


    # ============================================================
    # RESULTADOS
    # ============================================================

    logger.success(
        "Avaliação final concluída."
    )

    logger.info(
        f"Accuracy: "
        f"{results['accuracy']:.3f}"
    )

    logger.info(
        f"Precision: "
        f"{results['precision']:.3f}"
    )

    logger.info(
        f"Recall: "
        f"{results['recall']:.3f}"
    )

    logger.info(
        f"F1: "
        f"{results['f1']:.3f}"
    )

    logger.info(
        f"ROC AUC: "
        f"{results['roc_auc']:.3f}"
    )


    # ============================================================
    # FINALIZAÇÃO
    # ============================================================

    logger.success(
        "Pipeline Olist finalizado com sucesso!"
    )


if __name__ == "__main__":
    main()