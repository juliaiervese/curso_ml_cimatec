import numpy as np

from .config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    create_directories,
)
from .plots import plot_examples


def normalize_images(images):
    # Converte os pixels para float32 e normaliza de 0–255 para 0–1.
    return images.astype("float32") / 255.0


def main():
    # O processamento depende dos dados gerados por dataset.py.
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            "Execute primeiro o módulo dataset."
        )

    create_directories()

    # Abre o arquivo e o fecha automaticamente ao terminar o bloco.
    with np.load(RAW_DATA_FILE, allow_pickle=False) as raw:
        # Processa uma matriz por vez para reduzir o uso de memória.
        for name in ("X_train", "X_test", "y_train", "y_test"):
            values = raw[name]

            if name.startswith("X_"):
                # As matrizes X contêm imagens.
                values = normalize_images(values)
            else:
                # As matrizes y contêm rótulos.
                # Transforma o formato (N, 1) em (N,).
                values = values.flatten()

            # Salva cada matriz preparada em um arquivo próprio.
            np.save(
                PROCESSED_DATA_DIR / f"{name}.npy",
                values,
            )

            print(
                f"{name}: formato {values.shape}; "
                f"tipo {values.dtype}"
            )

            # Libera a referência ao array antes da próxima iteração.
            del values

    # mmap_mode permite acessar os arquivos sem carregar
    # todas as imagens na memória para gerar os nove exemplos.
    X_train = np.load(
        PROCESSED_DATA_DIR / "X_train.npy",
        mmap_mode="r",
    )

    y_train = np.load(
        PROCESSED_DATA_DIR / "y_train.npy",
        mmap_mode="r",
    )

    # Gera a figura com as nove primeiras imagens do treino.
    plot_examples(X_train[:9], y_train[:9])

    print(f"Dados preparados em: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()