"""Constrói a base de embeddings a partir das fotos em data/raw/autorizados.

Organização aceita:
    data/raw/autorizados/<nome_da_pessoa>/foto1.jpg   (recomendado)
    data/raw/autorizados/<nome_da_pessoa>.jpg
"""
import pickle
from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import typer

from reconhecimento_facial.config import (
    AUTORIZADOS_DIR,
    BASE_EMBEDDINGS,
    EXTENSOES,
    MODELO,
)
from reconhecimento_facial.features import extrair_rostos

app = typer.Typer(add_completion=False)


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES)


def ler_imagem(caminho: Path) -> np.ndarray | None:
    """Leitura segura para caminhos com acentos no Windows."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def nome_da_pessoa(caminho: Path) -> str:
    return caminho.parent.name if caminho.parent != AUTORIZADOS_DIR else caminho.stem


def construir_base() -> list[dict]:
    imagens = listar_imagens(AUTORIZADOS_DIR)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma foto encontrada em {AUTORIZADOS_DIR}")

    base = []
    for caminho in imagens:
        img = ler_imagem(caminho)
        if img is None:
            logger.warning(f"Não foi possível ler {caminho.name}")
            continue
        rostos = extrair_rostos(img)
        if not rostos:
            logger.warning(f"Nenhum rosto em {caminho.name}; ignorada")
            continue
        # se houver mais de um rosto, usa o maior (o principal da foto)
        rosto = max(rostos, key=lambda r: r["box"][2] * r["box"][3])
        base.append(
            {
                "pessoa": nome_da_pessoa(caminho),
                "arquivo": caminho.name,
                "embedding": rosto["embedding"],
            }
        )
        logger.info(f"OK: {nome_da_pessoa(caminho)} <- {caminho.name}")
    return base


def carregar_base() -> list[dict]:
    if not BASE_EMBEDDINGS.exists():
        raise FileNotFoundError(
            "Base não encontrada. Rode primeiro: python -m reconhecimento_facial.dataset"
        )
    with open(BASE_EMBEDDINGS, "rb") as f:
        return pickle.load(f)["registros"]


@app.command()
def main():
    """Gera models/base_autorizados.pkl."""
    logger.info(f"Modelo: {MODELO} | Fotos em: {AUTORIZADOS_DIR}")
    base = construir_base()
    if not base:
        raise typer.Exit("Nenhum rosto válido nas fotos autorizadas.")
    BASE_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    with open(BASE_EMBEDDINGS, "wb") as f:
        pickle.dump({"modelo": MODELO, "registros": base}, f)
    total, validas = {}, {}
    for c in listar_imagens(AUTORIZADOS_DIR):
        total[nome_da_pessoa(c)] = total.get(nome_da_pessoa(c), 0) + 1
    for r in base:
        validas[r["pessoa"]] = validas.get(r["pessoa"], 0) + 1
    for p in sorted(total):
        v = validas.get(p, 0)
        msg = f"{p}: {v}/{total[p]} fotos com rosto detectado"
        (logger.warning if v < total[p] else logger.info)(msg)
    pessoas = sorted({r["pessoa"] for r in base})
    logger.success(f"{len(base)} fotos / {len(pessoas)} pessoas salvas: {', '.join(pessoas)}")


if __name__ == "__main__":
    app()
