# DESENHO DAS BOUNDING BOXES (OpenCV)
#
# VERDE    + "ACESSO LIBERADO" + nome da pessoa
# VERMELHO + "ACESSO NEGADO"

import cv2

from module_reconhecimento_facial.config import (
    GREEN,
    LABEL_DENIED,
    LABEL_GRANTED,
    RED,
)


def box_style(granted: bool) -> tuple[tuple[int, int, int], str]:
    """Cor e texto da bounding box de acordo com a decisão de acesso."""
    return (GREEN, LABEL_GRANTED) if granted else (RED, LABEL_DENIED)


def draw_result(image, result: dict):
    """Desenha a bounding box e o texto de UM rosto na imagem (in place)."""
    x, y, w, h = result["x"], result["y"], result["w"], result["h"]
    color, label = box_style(result["granted"])

    # Espessura e fonte proporcionais ao tamanho da imagem
    scale = max(image.shape[:2]) / 800
    thickness = max(2, int(3 * scale))
    font_scale = max(0.6, 0.9 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    lines = [label]
    if result["granted"]:
        lines.append(result["name"].upper())

    # Faixa colorida com o texto, logo acima do rosto (ou dentro, se não couber)
    line_height = int(32 * font_scale)
    banner_height = line_height * len(lines) + 10
    top = y - banner_height if y - banner_height > 0 else y
    text_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in lines)

    cv2.rectangle(image, (x, top), (x + max(w, text_width + 12), top + banner_height), color, -1)

    for i, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (x + 6, top + line_height * (i + 1)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image


def draw_results(image, results: list[dict]):
    """Desenha todos os rostos reconhecidos em uma cópia da imagem."""
    output = image.copy()

    for result in results:
        draw_result(output, result)

    return output
