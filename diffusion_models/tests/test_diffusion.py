import pytest
import torch

from module_diffusion_models import config
from module_diffusion_models.diffusion import NoiseSchedule
from module_diffusion_models.modeling.predict import load_model, predict_eps, sample_ddim
from module_diffusion_models.modeling.train import train_step
from module_diffusion_models.modeling.unet import UNet, count_parameters

SCHEDULE = NoiseSchedule()


def small_unet():
    """U-Net pequena (CH = 8) para os testes rodarem rápido em CPU."""
    torch.manual_seed(0)
    return UNet(ch=8).eval()


# ---------------------------------------------------------------- estrutura do projeto


def test_pastas_do_projeto_existem():
    """A estrutura do Cookiecutter precisa existir para salvar dados, modelos e figuras."""
    pastas = [
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
    ]
    for pasta in pastas:
        assert pasta.exists(), f"Pasta não encontrada: {pasta}"


def test_caminhos_partem_da_raiz_do_projeto():
    assert config.DATA_DIR.parent == config.PROJ_ROOT
    assert config.MODELS_DIR.parent == config.PROJ_ROOT
    assert (config.PROJ_ROOT / "pyproject.toml").exists()


# ---------------------------------------------------------------- processo direto


def test_agendamento_de_ruido():
    """O sinal (ᾱ_t) começa perto de 1, só diminui e quase some no último passo."""
    ab = SCHEDULE.alphas_bar
    assert len(ab) == config.T
    assert ab[0] > 0.999
    assert torch.all(ab[1:] < ab[:-1])
    assert ab[-1] < 1e-4


def test_processo_direto_segue_a_formula():
    x0 = torch.rand(4, 1, 32, 32) * 2 - 1
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 10, 500, 999])

    xt = SCHEDULE.q_sample(x0, t, noise)

    ab = SCHEDULE.alphas_bar[t].view(-1, 1, 1, 1)
    assert torch.allclose(xt, ab.sqrt() * x0 + (1 - ab).sqrt() * noise)
    # em t = 0 a imagem quase não muda; em t = 999 sobra praticamente só o ruído
    assert torch.allclose(xt[0], x0[0], atol=0.05)
    assert torch.allclose(xt[3], noise[3], atol=0.05)


# ---------------------------------------------------------------- arquitetura


def test_unet_devolve_ruido_do_mesmo_tamanho_da_imagem():
    model = small_unet()
    x = torch.randn(3, 1, 32, 32)
    out = model(x, torch.tensor([0, 500, 999]), torch.tensor([1, 7, model.null_class]))
    assert out.shape == x.shape


def test_unet_do_experimento_tem_cerca_de_750_mil_parametros():
    assert 0.7e6 < count_parameters(UNet(ch=config.CH)) < 0.8e6


# ---------------------------------------------------------------- treino e geração


def test_passo_de_treino_atualiza_os_pesos():
    model = small_unet().train()
    ema = UNet(ch=8).eval()
    ema.load_state_dict(model.state_dict())
    antes = [p.clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = train_step(model, ema, SCHEDULE, torch.randn(8, 1, 32, 32), torch.randint(0, 10, (8,)), opt)

    assert loss > 0 and loss == loss  # positiva e não é NaN
    assert any(not torch.equal(a, p) for a, p in zip(antes, model.parameters()))


def test_guidance_combina_previsoes_com_e_sem_classe():
    """w = 1 é a previsão condicional pura; w = 0 é a incondicional."""
    model = small_unet()
    x, y = torch.randn(2, 1, 32, 32), torch.tensor([3, 8])
    t = torch.full((2,), 500)

    cond = model(x, t, y)
    incond = model(x, t, torch.full_like(y, model.null_class))

    assert torch.allclose(predict_eps(model, x, 500, y, guidance=1.0), cond, atol=1e-5)
    assert torch.allclose(predict_eps(model, x, 500, y, guidance=0.0), incond, atol=1e-5)


def test_ddim_e_deterministico_e_gera_imagens_validas():
    model = small_unet()
    y = torch.arange(4)

    a, frames = sample_ddim(model, SCHEDULE, y, steps=5, seed=123, save_every=1)
    b, _ = sample_ddim(model, SCHEDULE, y, steps=5, seed=123)

    assert a.shape == (4, 1, 32, 32)
    assert a.min() >= -1 and a.max() <= 1
    assert torch.equal(a, b)  # mesma semente -> mesma imagem
    assert len(frames) == 6  # 5 passos + imagem final


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="modelo treinado não encontrado")
def test_modelo_treinado_carrega_e_gera_digitos():
    model = load_model()
    imgs, _ = sample_ddim(model, SCHEDULE, torch.arange(10), steps=10, seed=0)
    assert imgs.shape == (10, 1, 32, 32)
    assert imgs.std() > 0.3  # não é uma imagem constante
