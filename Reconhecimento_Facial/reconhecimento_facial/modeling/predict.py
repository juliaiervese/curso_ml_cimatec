"""Inferência: fotos de teste (imagens) ou webcam."""
import csv
from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import typer

from reconhecimento_facial.config import (
    LIMIAR,
    RELATORIO_CSV,
    RESULTADOS_DIR,
    TESTE_DIR,
)
from reconhecimento_facial.dataset import carregar_base, ler_imagem, listar_imagens
from reconhecimento_facial.features import (
    decidir_acesso,
    distancia_cosseno,
    extrair_rostos,
)
from reconhecimento_facial.plots import desenhar_resultado

app = typer.Typer(add_completion=False)


def identificar(embedding: np.ndarray, base: list[dict]) -> tuple[str | None, float | None]:
    """Pessoa autorizada mais próxima e a distância (cosseno)."""
    if not base:
        return None, None
    melhor = min(base, key=lambda r: distancia_cosseno(embedding, r["embedding"]))
    return melhor["pessoa"], distancia_cosseno(embedding, melhor["embedding"])


def analisar(img: np.ndarray, base: list[dict]) -> list[dict]:
    resultados = []
    for rosto in extrair_rostos(img):
        pessoa, dist = identificar(rosto["embedding"], base)
        liberado = decidir_acesso(dist)
        resultados.append(
            {
                "box": rosto["box"],
                "pessoa": pessoa if liberado else None,
                "distancia": dist,
                "liberado": liberado,
            }
        )
    return resultados


def salvar_imagem(caminho: Path, img: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(caminho.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(caminho))


@app.command()
def imagens(pasta: Path = TESTE_DIR, saida: Path = RESULTADOS_DIR):
    """Processa todas as fotos da pasta de teste e salva as imagens anotadas."""
    base = carregar_base()
    fotos = listar_imagens(pasta)
    if not fotos:
        raise typer.Exit(f"Nenhuma foto de teste em {pasta}")

    linhas = []
    for caminho in fotos:
        img = ler_imagem(caminho)
        if img is None:
            logger.warning(f"Não foi possível ler {caminho.name}")
            continue
        resultados = analisar(img, base)
        if not resultados:
            logger.warning(f"{caminho.name}: nenhum rosto detectado")
            linhas.append([caminho.name, "", "sem rosto", ""])
        for r in resultados:
            desenhar_resultado(img, r)
            status = "liberado" if r["liberado"] else "negado"
            logger.info(f"{caminho.name}: acesso {status} (dist={r['distancia']:.3f})")
            linhas.append([caminho.name, r["pessoa"] or "", status, f"{r['distancia']:.4f}"])
        salvar_imagem(saida / f"{caminho.stem}_resultado{caminho.suffix}", img)

    RELATORIO_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RELATORIO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "pessoa", "status", "distancia"])
        w.writerows(linhas)
    logger.success(f"Imagens em {saida} | relatório em {RELATORIO_CSV} | limiar={LIMIAR}")


@app.command()
def webcam(camera: int = 0, a_cada: int = 5):
    """Reconhecimento em tempo real. Analisa 1 a cada N quadros. Tecle q para sair."""
    base = carregar_base()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise typer.Exit("Não foi possível abrir a webcam.")
    n, ultimos = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % a_cada == 0:
            ultimos = analisar(frame, base)
        n += 1
        for r in ultimos:
            desenhar_resultado(frame, r)
        cv2.imshow("Reconhecimento Facial (q para sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
