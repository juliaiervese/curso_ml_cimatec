import numpy as np

from .config import RAW_DATA_FILE, create_directories


def main():
    # Prepara as pastas que receberão os arquivos.
    create_directories()

    # Reutiliza os dados brutos caso já tenham sido salvos.
    if RAW_DATA_FILE.exists():
        print(f"Dados brutos já disponíveis em: {RAW_DATA_FILE}")
        return

    # A importação fica dentro da função.
    # Importar este módulo não inicia o carregamento da base.
    from tensorflow import keras

    # Carrega as imagens e os rótulos de treino e teste.
    # Na primeira utilização, o Keras baixa a CIFAR-10.
    (X_train, y_train), (X_test, y_test) = (
        keras.datasets.cifar10.load_data()
    )

    # Salva os quatro arrays em um arquivo compactado.
    # Nesta etapa, os pixels permanecem no intervalo original de 0 a 255.
    np.savez_compressed(
        RAW_DATA_FILE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Dados brutos salvos em: {RAW_DATA_FILE}")
    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")
    print(f"Formato de X_test: {X_test.shape}")
    print(f"Formato de y_test: {y_test.shape}")


# Executa main somente quando este módulo é chamado como programa.
if __name__ == "__main__":
    main()