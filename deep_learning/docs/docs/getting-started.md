Primeiros passos
================

1. Instale o [uv](https://docs.astral.sh/uv/) e rode `uv sync` na pasta `deep_learning/`.
2. Abra os notebooks da pasta `notebooks/` com o kernel do ambiente do projeto.
3. Para treinar a CNN do CIFAR-10 pelo pipeline modular:

    ```bash
    uv run python -m module_deep_learning.main
    ```

    Na primeira execução a base é baixada para `data/raw/cifar10.npz`. O melhor modelo é
    salvo em `models/melhor_modelo_cifar10.keras` e as métricas e gráficos em `reports/`.

4. Para rodar os testes:

    ```bash
    uv run pytest tests
    ```
