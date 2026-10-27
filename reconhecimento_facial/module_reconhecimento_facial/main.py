# SISTEMA DE CONTROLE DE ACESSO POR RECONHECIMENTO FACIAL
#
# Uso:
#   python -m module_reconhecimento_facial.main            # verifica as fotos de data/teste
#   python -m module_reconhecimento_facial.main --webcam   # verifica ao vivo pela webcam
#
# Fluxo:
#   1. cadastra as pessoas de data/autorizados (embeddings)
#   2. para cada foto de data/teste: detecta os rostos e compara com os autorizados
#   3. desenha a bounding box VERDE (ACESSO LIBERADO) ou VERMELHA (ACESSO NEGADO)
#   4. salva as imagens e um relatório CSV em outputs/

import argparse
import csv

import cv2
from loguru import logger

from module_reconhecimento_facial.config import OUTPUT_DIR, TEST_DIR
from module_reconhecimento_facial.database import build_database, list_images
from module_reconhecimento_facial.drawing import draw_results
from module_reconhecimento_facial.image_io import read_image, write_image
from module_reconhecimento_facial.recognizer import THRESHOLD, recognize


def process_test_images(database, test_dir=TEST_DIR, output_dir=OUTPUT_DIR):
    """Verifica todas as fotos da pasta de teste e salva o resultado."""
    images = list_images(test_dir)

    if not images:
        raise FileNotFoundError(f"Nenhuma foto encontrada em {test_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for image_path in images:
        image = read_image(image_path)

        if image is None:
            logger.warning(f"Não foi possível abrir {image_path.name}.")
            continue

        results = recognize(image, database)

        if not results:
            logger.warning(f"{image_path.name}: nenhum rosto detectado.")

        for result in results:
            status = "LIBERADO" if result["granted"] else "NEGADO"
            person = result["name"] if result["granted"] else "desconhecido"

            logger.info(
                f"{image_path.name}: ACESSO {status} | pessoa: {person} | "
                f"distância: {result['distance']:.3f} (limite {THRESHOLD:.2f})"
            )

            report.append(
                {
                    "imagem": image_path.name,
                    "acesso": status,
                    "pessoa": person,
                    "pessoa_mais_parecida": result["name"],
                    "distancia": round(result["distance"], 4),
                    "threshold": round(THRESHOLD, 4),
                }
            )

        output_path = output_dir / f"resultado_{image_path.stem}.jpg"
        write_image(output_path, draw_results(image, results))

    report_path = output_dir / "relatorio_acessos.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["imagem", "acesso", "pessoa", "pessoa_mais_parecida", "distancia", "threshold"],
        )
        writer.writeheader()
        writer.writerows(report)

    granted = sum(row["acesso"] == "LIBERADO" for row in report)
    logger.success(
        f"{len(report)} rosto(s) verificado(s): {granted} liberado(s), "
        f"{len(report) - granted} negado(s)."
    )
    logger.success(f"Imagens e relatório salvos em: {output_dir}")

    return report


def run_webcam(database, camera_index=0, every_n_frames=10):
    """
    Reconhecimento ao vivo. Pressione 'q' para sair.

    O reconhecimento é feito a cada N quadros (é a parte mais lenta);
    nos quadros intermediários, as últimas caixas continuam na tela.
    """
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    logger.info("Webcam aberta. Pressione 'q' para sair.")
    results, frame_count = [], 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_count % every_n_frames == 0:
            results = recognize(frame, database)

        cv2.imshow("Controle de acesso", draw_results(frame, results))
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Controle de acesso por reconhecimento facial.")
    parser.add_argument("--webcam", action="store_true", help="usa a webcam em vez de data/teste")
    args = parser.parse_args()

    logger.info("Cadastrando pessoas autorizadas...")
    database = build_database()

    if args.webcam:
        run_webcam(database)
    else:
        process_test_images(database)


if __name__ == "__main__":
    main()
