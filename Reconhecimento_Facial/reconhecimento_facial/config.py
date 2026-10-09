"""Configurações centrais: caminhos, modelo, limiar e cores."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Caminhos ---------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

AUTORIZADOS_DIR = RAW_DATA_DIR / "autorizados"  # fotos de quem pode entrar
TESTE_DIR = RAW_DATA_DIR / "teste"  # fotos para testar o sistema
RESULTADOS_DIR = PROCESSED_DATA_DIR / "resultados"  # imagens com bounding box
BASE_EMBEDDINGS = MODELS_DIR / "base_autorizados.pkl"
RELATORIO_CSV = REPORTS_DIR / "resultados.csv"

# --- Reconhecimento ---------------------------------------------------------
MODELO = os.getenv("FACE_MODEL", "Facenet512")
DETECTOR = os.getenv("FACE_DETECTOR", "retinaface")  # alternativas: mtcnn, ssd, opencv (opencv perde rostos de lado/inclinados)
LIMIAR = float(os.getenv("FACE_LIMIAR", "0.30"))  # distância cosseno máx. p/ liberar
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Visual (OpenCV usa BGR) -----------------------------------------------
COR_VERDE = (0, 255, 0)
COR_VERMELHA = (0, 0, 255)
TEXTO_LIBERADO = "acesso liberado"
TEXTO_NEGADO = "acesso negado"
