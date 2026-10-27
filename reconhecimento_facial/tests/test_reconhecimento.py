import numpy as np

from module_reconhecimento_facial.config import GREEN, LABEL_DENIED, LABEL_GRANTED, RED
from module_reconhecimento_facial.database import largest_face, person_name
from module_reconhecimento_facial.drawing import box_style, draw_results
from module_reconhecimento_facial.recognizer import THRESHOLD, match_face

DATABASE = [
    {"name": "bianca", "embedding": np.array([1.0, 0.0, 0.0])},
    {"name": "pedro", "embedding": np.array([0.0, 1.0, 0.0])},
]


def test_rosto_parecido_com_autorizado_libera_acesso():
    result = match_face([0.99, 0.05, 0.0], DATABASE)

    assert result["granted"]
    assert result["name"] == "bianca"
    assert result["distance"] < THRESHOLD


def test_rosto_desconhecido_nega_acesso():
    # Vetor ortogonal aos dois autorizados: distância de cosseno = 1
    result = match_face([0.0, 0.0, 1.0], DATABASE)

    assert not result["granted"]
    assert result["distance"] > THRESHOLD


def test_cores_e_textos_da_bounding_box():
    assert box_style(True) == (GREEN, LABEL_GRANTED)
    assert box_style(False) == (RED, LABEL_DENIED)
    assert LABEL_GRANTED == "ACESSO LIBERADO"
    assert LABEL_DENIED == "ACESSO NEGADO"


def test_desenho_pinta_a_caixa_com_a_cor_certa():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    results = [
        {"x": 50, "y": 200, "w": 100, "h": 100, "granted": True, "name": "bianca"},
        {"x": 250, "y": 200, "w": 100, "h": 100, "granted": False, "name": "pedro"},
    ]

    output = draw_results(image, results)

    # Borda inferior de cada caixa: verde no rosto liberado, vermelha no negado
    assert tuple(output[299, 100]) == GREEN
    assert tuple(output[299, 300]) == RED
    # A imagem original não é alterada
    assert image.sum() == 0


def test_nome_da_pessoa_vem_do_arquivo_ou_da_subpasta(tmp_path):
    assert person_name(tmp_path / "bianca_2.jpg", tmp_path) == "bianca"
    assert person_name(tmp_path / "pedro" / "foto1.jpg", tmp_path) == "pedro"


def test_cadastro_usa_o_maior_rosto():
    faces = [
        {"facial_area": {"w": 50, "h": 50}, "id": "pequeno"},
        {"facial_area": {"w": 200, "h": 180}, "id": "grande"},
    ]
    assert largest_face(faces)["id"] == "grande"


def test_le_e_salva_imagem_em_caminho_com_acento(tmp_path):
    from module_reconhecimento_facial.image_io import read_image, write_image

    folder = tmp_path / "Área de Trabalho"
    folder.mkdir()
    image = np.full((20, 30, 3), 120, dtype=np.uint8)

    write_image(folder / "foto.png", image)
    loaded = read_image(folder / "foto.png")

    assert loaded.shape == (20, 30, 3)
    assert (loaded == image).all()
