# Deep Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de Deep Learning da disciplina de Machine Learning (SENAI CIMATEC), com TensorFlow e Keras.

## Conteúdo

| Notebook / módulo | O que mostra |
|---|---|
| `notebooks/atividade_tensorflow.ipynb` | Neurônio artificial, pesos e bias, ReLU, sigmoid, forward pass, threshold e Binary Cross-Entropy (sem treino) |
| `notebooks/02_treinamento_redes_neurais_keras.ipynb` | Como a rede aprende: backpropagation com `tf.GradientTape`, learning rate, Keras `fit`, Adam × SGD, funções de ativação, MLP para risco de atraso na Olist, overfitting e regularização (Dropout, BatchNorm, Early Stopping) |
| `notebooks/Classificador_Imagens-v1.ipynb` e `-v2.ipynb` | CNN para classificar imagens do CIFAR-10 |
| `module_deep_learning/` | Pipeline modular da CNN do CIFAR-10: dados → features → split → modelo → treino → avaliação → inferência → gráficos |

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

```bash
uv sync
```

**Notebooks:** abra no VS Code/Jupyter com o kernel do ambiente do projeto.
O notebook `02_treinamento_redes_neurais_keras.ipynb` usa a base refinada do projeto Olist
(`../olist/data/interim/orders_dataset_refined.csv`), gerada por
`uv run python -m module_olist.main` dentro da pasta `olist/`.

**Pipeline da CNN (CIFAR-10):**

```bash
uv run python -m module_deep_learning.main
```

Na primeira execução a base CIFAR-10 é baixada para `data/raw/cifar10.npz`.

**Testes:**

```bash
uv run pytest tests
```

## Organização do projeto

```
├── Makefile           <- Comandos de conveniência (make requirements, make data, make train, make test)
├── README.md          <- Este arquivo
├── data               <- NÃO versionado (só a estrutura de pastas vai para o Git)
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw            <- cifar10.npz (baixado na primeira execução)
├── docs               <- Documentação mkdocs
├── models             <- Modelos treinados (.keras)
├── notebooks          <- Notebooks das aulas
├── pyproject.toml     <- Dependências do projeto (uv)
├── reports            <- Métricas, histórico de treino e figuras
├── tests              <- Testes com pytest
└── module_deep_learning
    ├── config.py      <- Caminhos e hiperparâmetros
    ├── dataset.py     <- Download e carga do CIFAR-10
    ├── features.py    <- Normalização e data augmentation
    ├── split.py       <- Separação treino/validação
    ├── main.py        <- Orquestra o pipeline completo
    ├── inference.py   <- Inferência de exemplo
    ├── plots.py       <- Gráficos
    └── modeling
        ├── model.py   <- Arquitetura da CNN
        ├── train.py   <- Treino com callbacks
        ├── evaluate.py
        └── predict.py
```
