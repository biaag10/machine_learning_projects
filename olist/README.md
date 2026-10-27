# Olist: risco de atraso de pedidos

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de Machine Learning da disciplina de Machine Learning (SENAI CIMATEC) usando o
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Problema de negócio

Identificar, **logo após a aprovação do pagamento** (`order_approved_at`), quais pedidos têm
maior risco de serem entregues **depois do prazo prometido**.

- **Alvo:** `is_late = 1` se a entrega real for depois da entrega estimada.
- **Instante da previsão:** aprovação do pagamento. Só entram como features informações já
  conhecidas nesse momento (sem *data leakage*: data de envio, data real de entrega e
  avaliação do cliente ficam de fora).
- **Desbalanceamento:** apenas ~8% dos pedidos atrasam, então as métricas principais são
  PR-AUC e F1, não a acurácia.

## Pipeline

```
data/raw (CSVs Olist)
   │  dataset.py   → filtra pedidos entregues, cria is_late, agrega itens, junta clientes
   ▼
base analítica (1 linha por pedido)
   │  features.py  → promised_days, purchase_month, purchase_weekday, purchase_hour
   ▼
data/interim/orders_dataset_refined.csv
   │  split.py     → treino/teste estratificado
   │  pipeline.py  → pré-processamento + Gradient Boosting / XGBoost / LightGBM
   │  modeling/cross_validation.py → seleção de modelo e threshold (StratifiedKFold)
   │  modeling/train.py / evaluate.py → treino final e avaliação no teste
   ▼
models/best_model.joblib + metadata.json
   │  inference.py → previsão para novos pedidos
   │  explain.py   → explicabilidade com SHAP
   │  active_learning.py → simulação de active learning
```

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

1. Baixe a base no Kaggle e coloque estes arquivos em `data/raw/`:
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_customers_dataset.csv`

2. Instale as dependências:

   ```bash
   uv sync
   ```

3. Rode as etapas:

   ```bash
   uv run python -m module_olist.main             # dados → features → CV → treino → avaliação
   uv run python -m module_olist.inference        # inferência com o modelo salvo
   uv run python -m module_olist.explain          # gráficos SHAP em reports/figures
   uv run python -m module_olist.active_learning  # curva de active learning
   ```

4. Testes:

   ```bash
   uv run pytest tests
   ```

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make train, make test...)
├── README.md          <- Este arquivo
├── README_olist_orders_dataset.md <- Dicionário da base de pedidos
├── data               <- NÃO versionado (só a estrutura de pastas vai para o Git)
│   ├── external       <- Dados de terceiros
│   ├── interim        <- Dados intermediários (base refinada)
│   ├── processed      <- Dados finais para modelagem
│   └── raw            <- CSVs originais da Olist (imutáveis)
├── docs               <- Documentação mkdocs
├── models             <- Modelo treinado, metadados e predições de exemplo
├── notebooks          <- Qualidade dos dados, validação com Pandera e EDA
├── pyproject.toml     <- Dependências e configuração do projeto (uv)
├── references         <- Materiais de apoio
├── reports            <- Relatórios gerados
│   └── figures        <- Gráficos (SHAP, active learning)
├── tests              <- Testes com pytest
└── module_olist       <- Código-fonte do projeto
    ├── config.py      <- Caminhos do projeto (RAW_DATA_DIR, MODELS_DIR...)
    ├── dataset.py     <- Carga, alvo e integração das tabelas
    ├── features.py    <- Engenharia de features
    ├── split.py       <- Separação treino/teste
    ├── pipeline.py    <- Pré-processamento + modelos
    ├── main.py        <- Orquestra o pipeline completo
    ├── inference.py   <- Inferência
    ├── explain.py     <- Explicabilidade (SHAP)
    ├── active_learning.py <- Active learning
    └── modeling
        ├── cross_validation.py
        ├── train.py
        ├── evaluate.py
        ├── predict.py
        └── interpret.py
```
