import argparse
import json

import numpy as np
from tensorflow import keras

from ..config import (
    PROCESSED_DATA_DIR,
    MODEL_FILE,
    HISTORY_FILE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    PATIENCE,
    SEED,
    create_directories,
)

from ..plots import plot_history
from .model import build_model


def main(epochs=EPOCHS):
    # Evita iniciar o treinamento com um número inválido de épocas.
    if epochs < 1:
        raise ValueError(
            "O número de épocas deve ser positivo."
        )

    # Arquivos preparados por features.py.
    files = [
        PROCESSED_DATA_DIR / f"{name}.npy"
        for name in ("X_train", "y_train")
    ]

    if not all(path.exists() for path in files):
        raise FileNotFoundError(
            "Execute primeiro os módulos dataset e features."
        )

    create_directories()

    # Configura as sementes utilizadas pelo Keras.
    keras.utils.set_random_seed(SEED)

    X_train = np.load(
        files[0],
        allow_pickle=False,
    )

    y_train = np.load(
        files[1],
        allow_pickle=False,
    )

    # Cada execução constrói um modelo novo.
    # Este script não retoma automaticamente um treinamento anterior.
    model = build_model()

    # Exibe as camadas e a quantidade de parâmetros.
    model.summary()

    # Interrompe o treino quando val_loss deixa de melhorar
    # por três épocas consecutivas.
    # Ao finalizar, recupera os pesos da melhor época.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    # Reserva 20% dos dados fornecidos para validação.
    # Os dados de teste não participam desta etapa.
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
    )

    # Salva o modelo para utilização em outro processo.
    # Uma nova execução substitui o modelo salvo anteriormente.
    model.save(MODEL_FILE)

    # Salva as métricas registradas em cada época.
    with HISTORY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history.history,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Gera os gráficos de acurácia e perda.
    plot_history(history.history)

    print(f"Modelo salvo em: {MODEL_FILE}")
    print(f"Histórico salvo em: {HISTORY_FILE}")


if __name__ == "__main__":
    # Permite informar o número de épocas pelo terminal.
    parser = argparse.ArgumentParser(
        description="Treinar o classificador CIFAR-10"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    args = parser.parse_args()

    main(epochs=args.epochs)