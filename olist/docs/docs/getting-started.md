Primeiros passos
================

1. Instale o [uv](https://docs.astral.sh/uv/) e rode `uv sync` na pasta `olist/`.
2. Baixe o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   e copie `olist_orders_dataset.csv`, `olist_order_items_dataset.csv` e
   `olist_customers_dataset.csv` para `data/raw/`.
3. Rode o pipeline completo:

    ```bash
    uv run python -m module_olist.main
    ```

    Ele gera `data/interim/orders_dataset_refined.csv`, seleciona o melhor modelo com
    validação cruzada e salva `models/best_model.joblib` e `models/metadata.json`.

4. Depois rode inferência, explicabilidade e active learning:

    ```bash
    uv run python -m module_olist.inference
    uv run python -m module_olist.explain
    uv run python -m module_olist.active_learning
    ```
