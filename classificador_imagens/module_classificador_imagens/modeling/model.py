from tensorflow import keras

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)

from ..config import INPUT_SHAPE, CLASS_NAMES


def build_model():
    # Define a entrada para imagens RGB de 32 × 32 pixels.
    model = keras.Sequential([
        Input(shape=INPUT_SHAPE)
    ])

    # Mantém os três blocos convolucionais do notebook.
    # Cada par informa o número de filtros e a taxa de dropout.
    for filters, dropout_rate in (
        (32, 0.25),
        (64, 0.25),
        (128, 0.30),
    ):
        # Primeira convolução do bloco.
        # padding="same" preserva as dimensões espaciais nesta camada.
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )

        model.add(BatchNormalization())

        # Segunda convolução do bloco.
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )

        model.add(BatchNormalization())

        # Reduz as dimensões espaciais dos mapas de características.
        model.add(
            MaxPooling2D(pool_size=(2, 2))
        )

        # Desativa parte das unidades durante o treinamento.
        model.add(Dropout(dropout_rate))

    # Transforma os mapas de características em um vetor.
    model.add(Flatten())

    # Combina as características aprendidas pelos blocos convolucionais.
    model.add(
        Dense(256, activation="relu")
    )

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Produz uma probabilidade para cada uma das dez classes.
    model.add(
        Dense(
            len(CLASS_NAMES),
            activation="softmax",
        )
    )

    # Os rótulos são números inteiros, por isso a loss é sparse.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Retorna a rede construída e compilada, ainda sem treinamento.
    return model