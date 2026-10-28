# Diffusion Models

## Descrição

Projeto em grupo da disciplina de IA Generativa (SENAI CIMATEC): um modelo de difusão (DDPM)
condicionado à classe, treinado do zero no MNIST, e a mesma técnica em escala real com o
Stable Diffusion (SD-Turbo).

## Comandos

O `Makefile` concentra os comandos principais:

- `make requirements`: instala as dependências com uv;
- `make data`: baixa o MNIST em `data/raw/`;
- `make train`: treina o DDPM do zero e salva `models/ddpm_mnist_ema.pt`;
- `make predict`: gera dígitos com o modelo treinado em `reports/figures/`;
- `make test`: roda os testes.

O roteiro de fala da apresentação está em [roteiro_de_fala.md](roteiro_de_fala.md).
