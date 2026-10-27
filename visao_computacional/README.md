# Visão Computacional

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de Visão Computacional da disciplina de Machine Learning (SENAI CIMATEC), com
OpenCV, TensorFlow e Keras.

## Objetivo

Aplicar redes neurais a **dados não estruturados (imagens)**:

- fundamentos de imagens digitais: pixels, canais (RGB), resolução e normalização;
- Redes Neurais Convolucionais (CNN): convolução, filtros, pooling, flatten e camadas densas;
- classificação de imagens com CNN.

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

```bash
uv sync
```

**Notebooks:** abra os notebooks da pasta `notebooks/` no VS Code/Jupyter com o kernel do
ambiente do projeto.

**Testes:**

```bash
uv run pytest tests
```

## Organização do projeto

```
├── LICENSE            <- Licença MIT
├── Makefile           <- Comandos de conveniência (make requirements, make data, make test)
├── README.md          <- Este arquivo
├── data               <- NÃO versionado (só a estrutura de pastas vai para o Git)
│   ├── external       <- Dados de terceiros
│   ├── interim        <- Dados intermediários
│   ├── processed      <- Dados prontos para o modelo
│   └── raw            <- Imagens e bases originais
├── docs               <- Documentação mkdocs
├── models             <- Modelos treinados (.keras)
├── notebooks          <- Notebooks das aulas
├── pyproject.toml     <- Dependências do projeto (uv)
├── references         <- Materiais de apoio
├── reports            <- Métricas e relatórios gerados
│   └── figures        <- Gráficos e imagens geradas
├── tests              <- Testes com pytest
└── module_visao_computacional   <- Código-fonte do projeto
    ├── config.py      <- Caminhos do projeto (RAW_DATA_DIR, MODELS_DIR...)
    ├── dataset.py     <- Carga das imagens
    ├── features.py    <- Pré-processamento das imagens
    ├── plots.py       <- Visualizações
    └── modeling
        ├── train.py   <- Treino dos modelos
        └── predict.py <- Inferência
```
