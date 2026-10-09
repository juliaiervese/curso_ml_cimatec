"""Desenho das bounding boxes (verde = liberado, vermelho = negado)."""
import cv2
import numpy as np

from reconhecimento_facial.config import (
    COR_VERDE,
    COR_VERMELHA,
    TEXTO_LIBERADO,
    TEXTO_NEGADO,
)
from reconhecimento_facial.features import remover_acentos

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def desenhar_resultado(img: np.ndarray, r: dict) -> None:
    """Desenha a caixa e o texto de um resultado (modifica img in-place)."""
    x, y, w, h = r["box"]
    liberado = r["liberado"]
    cor = COR_VERDE if liberado else COR_VERMELHA
    texto = TEXTO_LIBERADO if liberado else TEXTO_NEGADO
    cor_texto = (0, 0, 0) if liberado else (255, 255, 255)

    h_img, w_img = img.shape[:2]
    espessura = max(2, round(max(h_img, w_img) / 400))
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, espessura)

    # Tamanho do texto proporcional à imagem, reduzido até caber na largura
    escala = max(0.4, max(h_img, w_img) / 1200)
    esp_txt = 2
    (tw, th), base = cv2.getTextSize(texto, FONTE, escala, esp_txt)
    while tw + 10 > w_img and escala > 0.25:
        escala -= 0.05
        esp_txt = 1 if escala < 0.5 else 2
        (tw, th), base = cv2.getTextSize(texto, FONTE, escala, esp_txt)

    # Rótulo acima da caixa, sem sair dos limites da imagem
    x_txt = int(min(max(x, 0), max(w_img - tw - 10, 0)))
    topo = max(y - th - base - 8, 0)
    cv2.rectangle(img, (x_txt, topo), (x_txt + tw + 10, topo + th + base + 8), cor, -1)
    cv2.putText(
        img, texto, (x_txt + 5, topo + th + 4), FONTE, escala, cor_texto, esp_txt, cv2.LINE_AA
    )

    if liberado and r.get("pessoa"):
        nome = f"{remover_acentos(r['pessoa'])} ({r['distancia']:.2f})"
        (nw, _), _ = cv2.getTextSize(nome, FONTE, escala, esp_txt)
        x_nome = int(min(max(x, 0), max(w_img - nw - 4, 0)))
        y_nome = int(min(y + h + th + 8, h_img - 4))
        cv2.putText(img, nome, (x_nome, y_nome), FONTE, escala, cor, esp_txt, cv2.LINE_AA)
