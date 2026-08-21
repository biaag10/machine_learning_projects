# main.py -> orquestrador: vai localizar os CSVs, 
# chamar load_dataset(), 
# passar os DataFrames para create_dataset(), 
# depois aplicar create_features() 
# e salvar o resultado dentro de data/interim.

from pathlib import Path

from loguru import logger

from module_olist.dataset import load_dataset, create_dataset, save_dataset
from module_olist.features import create_features

def main():
    # DEFINIÇÃO DOS CAMINHOS
    
    # Raiz do projeto Olist
    ROOT = Path(__file__).resolve().parents[1]
    
    # Diretórios de dados
    DATA_RAW = ROOT / "data" / "raw"
    DATA_INTERIM = ROOT / "data" / "interim"
    
    # Cria a pasta interim caso ela ainda não exista
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    
    # Arquivos de entrada
    orders_path = DATA_RAW / "olist_orders_dataset.csv"
    items_path = DATA_RAW / "olist_order_items_dataset.csv"
    customers_path = DATA_RAW / "olist_customers_dataset.csv"
    
    # Arquivo de saída
    output_path = DATA_INTERIM / "olist_orders_features.csv"

    logger.info("Iniciando pipeline Olist...")
    
    # LEITURA DOS DATAFRAMES

    logger.info("Carregando datasets...")

    orders, items, customers = load_dataset(
        orders_path=orders_path,
        items_path=items_path,
        customers_path=customers_path,
    )

    logger.info(f"Orders carregado: {orders.shape}")
    logger.info(f"Items carregado: {items.shape}")
    logger.info(f"Customers carregado: {customers.shape}")
    
     # CRIAÇÃO DA BASE ANALÍTICA

    logger.info("Criando dataset analítico...")

    data = create_dataset(
        orders=orders,
        items=items,
        customers=customers,
    )

    logger.info(f"Dataset analítico criado: {data.shape}")
    
    # CRIAÇÃO DAS FEATURES
    
    logger.info("Criando features...")

    data = create_features(data)

    logger.info(f"Dataset com features: {data.shape}")
    
    # SALVAMENTO NA PASTA INTERIM
    
    logger.info("Salvando dataset intermediário...")

    save_dataset(
        dataset=data,
        output_path=output_path,
    )

    logger.success(
        f"Pipeline finalizado com sucesso. "
        f"Arquivo salvo em: {output_path}"
    )
    
    pass




if __name__ == "__main__":
    main()   