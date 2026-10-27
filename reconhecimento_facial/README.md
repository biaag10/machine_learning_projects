# Reconhecimento Facial: controle de acesso

Projeto da disciplina de Machine Learning (SENAI CIMATEC): um sistema de controle de acesso
que reconhece rostos com **OpenCV** e **DeepFace**.

- Rosto de uma pessoa **autorizada** → bounding box **VERDE** com o texto **ACESSO LIBERADO** e o nome.
- Rosto **desconhecido** → bounding box **VERMELHA** com o texto **ACESSO NEGADO**.

## Como funciona

```
data/autorizados/*.jpg ──► DeepFace.represent ──► embedding (512 números) de cada autorizado
                              (detector RetinaFace    │
                               + modelo Facenet512)   │
                                                      ▼
data/teste/*.jpg ──► detecta TODOS os rostos ──► embedding de cada rosto
                                                      │
                                                      ▼
                       distância de cosseno até cada autorizado
                                                      │
                     menor distância <= 0,30 ?  ──► sim: VERDE  "ACESSO LIBERADO" + nome
                                                └─► não: VERMELHO "ACESSO NEGADO"
                                                      │
                                                      ▼
                    outputs/resultado_<foto>.jpg  +  outputs/relatorio_acessos.csv
```

- **Detecção (RetinaFace):** o detector RetinaFace, usado pelo DeepFace, encontra onde estão os rostos
  (x, y, largura, altura). Ele foi escolhido no lugar do Haar Cascade do OpenCV porque, nos testes, o Haar
  Cascade marcou o peito de uma pessoa como rosto e não achou rostos menores.
- **Reconhecimento (DeepFace):** o modelo **Facenet512** transforma cada rosto em um vetor de 512 números
  (*embedding*). Fotos da mesma pessoa geram vetores próximos.
- **Decisão:** se a distância de cosseno até algum autorizado for menor que o limite calibrado pelo
  DeepFace para o Facenet512 (0,30), o acesso é liberado.
- **Desenho (OpenCV):** `cv2.rectangle` e `cv2.putText` desenham a caixa e o texto.

## Estrutura de pastas

```
reconhecimento_facial/
├── README.md
├── pyproject.toml                  <- Dependências (uv)
├── data/
│   ├── autorizados/                <- Fotos das pessoas AUTORIZADAS a usar o sistema
│   └── teste/                      <- Fotos que serão verificadas
├── outputs/                        <- Imagens com as bounding boxes + relatório CSV
├── module_reconhecimento_facial/
│   ├── config.py                   <- Caminhos, modelo, detector, cores e textos
│   ├── database.py                 <- Cadastro (embeddings) das pessoas autorizadas
│   ├── recognizer.py               <- Detecção dos rostos e decisão de acesso
│   ├── drawing.py                  <- Bounding boxes verde/vermelha com OpenCV
│   ├── image_io.py                 <- Leitura/gravação de imagens (aceita acentos no caminho)
│   └── main.py                     <- Executa o sistema (fotos de teste ou webcam)
└── tests/
    └── test_reconhecimento.py      <- Testes da decisão de acesso e do desenho
```

## Como usar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

1. Instale as dependências:

   ```bash
   uv sync
   ```

2. Coloque as fotos das pessoas autorizadas em `data/autorizados/`. O nome do arquivo é o
   nome da pessoa (tudo antes do primeiro `_`):

   ```
   data/autorizados/bianca_1.jpg
   data/autorizados/bianca_2.jpg   <- mais de uma foto por pessoa melhora o reconhecimento
   ```

   Também é possível usar uma subpasta por pessoa: `data/autorizados/bianca/foto1.jpg`.

3. Coloque as fotos a verificar em `data/teste/` (de pessoas autorizadas e não autorizadas).

4. Execute:

   ```bash
   uv run python -m module_reconhecimento_facial.main
   ```

   As imagens com as bounding boxes ficam em `outputs/` e o resumo em `outputs/relatorio_acessos.csv`.

5. (Opcional) Reconhecimento ao vivo pela webcam (pressione `q` para sair):

   ```bash
   uv run python -m module_reconhecimento_facial.main --webcam
   ```

Na primeira execução, o DeepFace baixa os pesos do Facenet512 (cerca de 90 MB) e do RetinaFace
(cerca de 120 MB) para `~/.deepface/weights`.

> **Windows com acento no caminho** (ex.: `Área de Trabalho`): algumas partes do OpenCV não abrem
> arquivos em caminhos com acento. As fotos já são lidas de um jeito que evita isso
> (`image_io.py`), mas o ambiente virtual deve ficar em uma pasta sem acento:
>
> ```powershell
> $env:UV_PROJECT_ENVIRONMENT = "D:\venvs\reconhecimento_facial"
> uv sync
> ```

## Testes

```bash
uv run pytest tests
```

## Limitações

- Com **uma única foto** por pessoa, uma foto de outro ângulo ou iluminação pode ficar logo acima do
  limite e ser negada (falso negativo). Cadastrar 2 ou 3 fotos por pessoa reduz esse problema.
- O reconhecimento funciona melhor com rostos **de frente**, bem iluminados e **sem óculos escuros**
  (os olhos são uma das regiões mais importantes para o embedding).
- O sistema não tem **anti-spoofing**: uma foto impressa de um autorizado poderia liberar o acesso.
- Fotos de rosto são **dados pessoais**: use apenas imagens de pessoas que autorizaram o uso.
