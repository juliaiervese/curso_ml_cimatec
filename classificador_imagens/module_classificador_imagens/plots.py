import matplotlib

# Permite salvar as figuras sem abrir janelas gráficas.
# Deve ser configurado antes de importar pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import (
    CLASS_NAMES,
    FIGURES_DIR,
    create_directories,
)


def save_figure(fig, filename):
    # Garante que a pasta de figuras existe.
    create_directories()

    # Ajusta os espaços, reservando uma área para o título superior.
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Salva a figura como arquivo de imagem.
    fig.savefig(FIGURES_DIR / filename, dpi=150)

    # Fecha a figura para liberar seus recursos.
    plt.close(fig)


def plot_examples(images, labels):
    # Cria uma grade com nove posições.
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.axis("off")

        # Permite utilizar a função mesmo com menos de nove imagens.
        if i < len(images):
            ax.imshow(images[i])

            # Converte o rótulo numérico para o nome da classe.
            ax.set_title(CLASS_NAMES[int(labels[i])])

    fig.suptitle("Exemplos da base CIFAR-10")

    save_figure(fig, "exemplos_cifar10.png")


def plot_history(history):
    # O argumento history recebe o dicionário history.history do Keras.
    # Gera uma figura para acurácia e outra para perda.
    for metric, title, ylabel, filename in (
        (
            "accuracy",
            "Acurácia por época",
            "Acurácia",
            "acuracia.png",
        ),
        (
            "loss",
            "Loss por época",
            "Loss",
            "loss.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Numera as épocas começando em 1.
        epochs = range(1, len(history[metric]) + 1)

        ax.plot(
            epochs,
            history[metric],
            label="Treino",
        )

        ax.plot(
            epochs,
            history[f"val_{metric}"],
            label="Validação",
        )

        ax.set(
            title=title,
            xlabel="Época",
            ylabel=ylabel,
        )

        ax.legend()

        save_figure(fig, filename)


def plot_predictions(images, real_labels, predicted_labels):
    # Cria a grade para comparar classes reais e previstas.
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.axis("off")

        if i < len(images):
            ax.imshow(images[i])

            real = CLASS_NAMES[int(real_labels[i])]
            predicted = CLASS_NAMES[int(predicted_labels[i])]

            ax.set_title(
                f"Real: {real}\nPredição: {predicted}"
            )

    fig.suptitle(
        "Previsões em imagens do conjunto de teste"
    )

    save_figure(fig, "predicoes_teste.png")