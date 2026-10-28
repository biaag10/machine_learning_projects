# machine_learning_projects

Repositório com os projetos da disciplina de **Machine Learning** da **Universidade SENAI CIMATEC**
(Prof. Leonardo Cardia), desenvolvidos por **Bianca Andrade Galvão** ([@biaag10](https://github.com/biaag10)).

Os projetos acompanham a disciplina em ordem: do Machine Learning clássico com dados tabulares
até redes neurais, visão computacional e IA generativa. Cada pasta é um projeto independente, com
seu próprio ambiente, README e instruções de execução.

## Projetos

| Projeto                                            | Tema                             | O que foi feito                                                                                                                                                                                                                           | Principais tecnologias                                 |
| -------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [`olist/`](olist/)                                 | ML com dados tabulares           | Previsão do **risco de atraso de pedidos** do e-commerce Olist no momento da aprovação do pagamento: qualidade dos dados, EDA, engenharia de features sem _data leakage_, validação cruzada, explicabilidade com SHAP e _active learning_ | pandas, Pandera, scikit-learn, XGBoost, LightGBM, SHAP |
| [`deep_learning/`](deep_learning/)                 | Redes neurais                    | Neurônio artificial, _forward pass_, _backpropagation_ com `GradientTape`, otimizadores, regularização (Dropout, BatchNorm, Early Stopping), MLP para a base Olist e pipeline modular de uma CNN no CIFAR-10                              | TensorFlow, Keras                                      |
| [`visao_computacional/`](visao_computacional/)     | Visão computacional              | Fundamentos de imagens digitais e **classificação de imagens com CNN** no CIFAR-10, com data augmentation, matriz de confusão e mapas de características                                                                                  | OpenCV, TensorFlow, Keras                              |
| [`reconhecimento_facial/`](reconhecimento_facial/) | Reconhecimento facial            | Sistema de **controle de acesso**: caixa verde **ACESSO LIBERADO** para pessoas autorizadas e vermelha **ACESSO NEGADO** para desconhecidos, por foto ou webcam                                                                           | DeepFace (Facenet512, RetinaFace), OpenCV              |
| [`diffusion_models/`](diffusion_models/)           | IA generativa (projeto em grupo) | **Modelo de difusão (DDPM) implementado do zero**, treinado no MNIST para gerar o dígito pedido, com DDIM e _Classifier-Free Guidance_, avaliação com FID e demonstração com Stable Diffusion                                             | PyTorch, torchmetrics, Gradio, diffusers               |

## Destaques

| Projeto               | Resultado                                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Olist                 | XGBoost com **ROC-AUC 0,765** e **PR-AUC 0,264** no teste, contra 0,081 do baseline (só ~8% dos pedidos atrasam)             |
| Deep Learning         | MLP na base Olist com **ROC-AUC 0,752** e **PR-AUC 0,235**                                                                   |
| Visão Computacional   | CNN melhorada com **86,6% de acurácia** no teste do CIFAR-10, contra 64,8% da CNN base                                       |
| Reconhecimento Facial | Rostos autorizados liberados com distância de cosseno bem abaixo do limite de 0,30 e desconhecidos negados                   |
| Diffusion Models      | **97,2%** dos dígitos gerados reconhecidos como a classe pedida; DDIM com 50 passos **~19× mais rápido** que o DDPM com 1000 |

O projeto **Diffusion Models** foi feito em grupo com **Rafael Athaliba** e **Pedro Lima** (disciplina de
IA Generativa) e pode ser aberto direto no Google Colab pelo botão no
[README do projeto](diffusion_models/README.md).

## Estrutura do repositório

```
machine_learning_projects/
├── README.md                 <- Este arquivo
├── olist/                    <- ML tabular: risco de atraso de pedidos
├── deep_learning/            <- Redes neurais com TensorFlow/Keras
├── visao_computacional/      <- Classificação de imagens com CNN
├── reconhecimento_facial/    <- Controle de acesso com DeepFace
└── diffusion_models/         <- DDPM do zero + Stable Diffusion (projeto em grupo)
```

Os projetos seguem o padrão [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
(o `reconhecimento_facial/` usa uma versão enxuta, com `data/`, `outputs/`, `tests/` e o módulo):

```
<projeto>/
├── README.md          <- Explicação do projeto e de como executar
├── pyproject.toml     <- Dependências (uv)
├── Makefile           <- Comandos de conveniência (make requirements, make test, ...)
├── data/              <- Dados (não versionados; só a estrutura de pastas vai para o Git)
├── models/            <- Modelos treinados
├── notebooks/         <- Notebooks das aulas e experimentos
├── reports/figures/   <- Figuras e resultados
├── tests/             <- Testes com pytest
└── module_<projeto>/  <- Código-fonte do projeto, organizado em módulos
```

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

Cada projeto tem o seu próprio ambiente. Entre na pasta do projeto e rode:

```bash
cd <projeto>
uv sync              # cria o ambiente e instala as dependências
uv run pytest tests  # roda os testes
```

As instruções específicas (bases de dados, treino, notebooks) estão no README de cada projeto.

> **Windows com acento no caminho** (ex.: `Área de Trabalho`): algumas bibliotecas (OpenCV, TensorFlow)
> não abrem arquivos em caminhos com acento. Nesse caso, crie o ambiente em uma pasta sem acento:
> `$env:UV_PROJECT_ENVIRONMENT = "D:\venvs\<projeto>"` antes do `uv sync`.

## Padrão de commits

As mensagens de commit seguem o formato:

```
<tipo>(<nome_projeto>): <descrição>
```

Exemplo: `feat(olist): adiciona o SHAP Explainability.`

| Tipo       | Quando usar                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `feat`     | Inclui um novo recurso (relaciona-se com o MINOR do versionamento semântico)                                     |
| `fix`      | Soluciona um problema, um _bug fix_ (relaciona-se com o PATCH do versionamento semântico)                        |
| `docs`     | Mudanças na documentação, como o README (não inclui alterações em código)                                        |
| `test`     | Cria, altera ou exclui testes unitários (não inclui alterações em código)                                        |
| `build`    | Modificações em arquivos de build e dependências                                                                 |
| `perf`     | Alterações de código relacionadas a performance                                                                  |
| `style`    | Formatação de código, ponto e vírgula, espaços em branco, lint... (não inclui alterações em código)              |
| `refactor` | Refatorações que não alteram a funcionalidade, por exemplo após um code review                                   |
| `chore`    | Tarefas de build, configurações e pacotes, como adicionar algo ao `.gitignore` (não inclui alterações em código) |
| `ci`       | Mudanças relacionadas à integração contínua (_continuous integration_)                                           |
| `raw`      | Mudanças em arquivos de configuração, dados, features e parâmetros                                               |
| `cleanup`  | Remove código comentado, trechos desnecessários ou qualquer outra limpeza que melhore a legibilidade             |
| `remove`   | Exclui arquivos, diretórios ou funcionalidades obsoletas ou não utilizadas                                       |
