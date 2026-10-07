from pathlib import Path

# Localiza a raiz do projeto a partir deste arquivo.
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Pastas que armazenam os dados, o modelo e os resultados.
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Caminhos dos arquivos gerados pelas etapas.
RAW_DATA_FILE = RAW_DATA_DIR / "cifar10.npz"
MODEL_FILE = MODELS_DIR / "classificador_cifar10.keras"
HISTORY_FILE = REPORTS_DIR / "historico.json"
METRICS_FILE = REPORTS_DIR / "metricas_teste.json"

# A ordem das classes corresponde aos rótulos numéricos da CIFAR-10.
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]

# As imagens possuem 32 × 32 pixels e três canais de cor.
INPUT_SHAPE = (32, 32, 3)

# Parâmetros utilizados no treinamento.
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
PATIENCE = 3

# Reduz as variações aleatórias entre execuções.
SEED = 42


def create_directories():
    # Cria as pastas de saída, incluindo suas pastas superiores.
    # exist_ok=True evita erro quando a pasta já existe.
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)