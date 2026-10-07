import json

import numpy as np
from tensorflow import keras

from ..config import (
    PROCESSED_DATA_DIR,
    MODEL_FILE,
    METRICS_FILE,
    BATCH_SIZE,
    create_directories,
)

from ..plots import plot_predictions


def main():
    # A avaliação depende de um modelo previamente treinado.
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Execute primeiro o módulo modeling.train."
        )

    # Arquivos de teste preparados por features.py.
    files = [
        PROCESSED_DATA_DIR / f"{name}.npy"
        for name in ("X_test", "y_test")
    ]

    if not all(path.exists() for path in files):
        raise FileNotFoundError(
            "Execute primeiro os módulos dataset e features."
        )

    create_directories()

    X_test = np.load(
        files[0],
        allow_pickle=False,
    )

    y_test = np.load(
        files[1],
        allow_pickle=False,
    )

    # Carrega o modelo salvo sem realizar um novo treinamento.
    model = keras.models.load_model(MODEL_FILE)

    # Calcula perda e acurácia em todas as imagens do conjunto de teste.
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
    )

    # Exibe a acurácia em formato percentual.
    print(f"Acurácia no teste: {test_accuracy:.2%}")
    print(f"Loss no teste: {test_loss:.4f}")

    # Salva as métricas da avaliação final.
    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "accuracy": float(test_accuracy),
                "loss": float(test_loss),
            },
            file,
            indent=2,
        )

    # Gera probabilidades para as nove primeiras imagens de teste.
    probabilities = model.predict(X_test[:9])

    # Seleciona a classe com maior probabilidade em cada imagem.
    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    # Salva a figura comparando as classes reais e previstas.
    plot_predictions(
        X_test[:9],
        y_test[:9],
        predicted_labels,
    )

    print(f"Métricas salvas em: {METRICS_FILE}")


if __name__ == "__main__":
    main()