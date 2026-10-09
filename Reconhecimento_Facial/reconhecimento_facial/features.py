"""Extração de rostos/embeddings e regras de decisão."""
import unicodedata

from loguru import logger
import numpy as np

from reconhecimento_facial.config import DETECTOR, LIMIAR, MODELO


def distancia_cosseno(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def decidir_acesso(distancia, limiar: float = LIMIAR) -> bool:
    """True = acesso liberado. Sem distância (ninguém na base) = negado."""
    return distancia is not None and distancia <= limiar


def remover_acentos(texto: str) -> str:
    """cv2.putText não desenha acentos; normaliza para ASCII."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def extrair_rostos(imagem_bgr: np.ndarray) -> list[dict]:
    """Detecta rostos e devolve [{'embedding': ndarray, 'box': (x, y, w, h)}]."""
    from deepface import DeepFace  # import tardio: DeepFace/TensorFlow são pesados

    try:
        resultados = DeepFace.represent(
            img_path=imagem_bgr,
            model_name=MODELO,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
    except ValueError as e:  # nenhum rosto detectado
        logger.debug(f"Sem rosto: {e}")
        return []

    rostos = []
    for r in resultados:
        a = r["facial_area"]
        rostos.append(
            {"embedding": np.array(r["embedding"]), "box": (a["x"], a["y"], a["w"], a["h"])}
        )
    return rostos
