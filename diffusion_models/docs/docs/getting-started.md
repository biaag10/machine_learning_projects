Primeiros passos
================

1. Instale o [uv](https://docs.astral.sh/uv/) e rode `uv sync` na pasta `diffusion_models/`.
2. Rode os testes:

    ```bash
    uv run pytest tests
    ```

3. Gere dígitos com o modelo já treinado:

    ```bash
    uv run python -m module_diffusion_models.modeling.predict --digito 7
    ```

4. Abra `notebooks/Diffusion_Models_MNIST_SD.ipynb` com o kernel do ambiente do projeto
   (ou pelo botão "Open in Colab" do README).
