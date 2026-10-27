from module_olist import config


def test_pastas_do_projeto_existem():
    """A estrutura do Cookiecutter precisa existir para o pipeline salvar os arquivos."""
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
    """Nenhum caminho deve depender da máquina de quem roda o projeto."""
    assert config.DATA_DIR.parent == config.PROJ_ROOT
    assert config.MODELS_DIR.parent == config.PROJ_ROOT
    assert (config.PROJ_ROOT / "pyproject.toml").exists()
