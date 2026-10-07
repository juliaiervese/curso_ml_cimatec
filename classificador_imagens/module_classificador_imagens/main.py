import argparse
import json
import os

from module_classificador_imagens import dataset, features

from module_classificador_imagens.config import (
    EPOCHS,
    FIGURES_DIR,
    HISTORY_FILE,
    METRICS_FILE,
    MODEL_FILE,
)

from module_classificador_imagens.modeling import predict, train
from module_classificador_imagens.plots import plot_history


def show_results():
    # Recria os gráficos de acurácia e perda usando o histórico salvo.
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(encoding="utf-8") as file:
            history = json.load(file)

        plot_history(history)

    # Carrega as métricas calculadas durante a avaliação.
    with METRICS_FILE.open(encoding="utf-8") as file:
        metrics = json.load(file)

    # Centraliza o resumo dos resultados no terminal.
    print("\nResultados da avaliação")
    print(f"Acurácia: {metrics['accuracy']:.2%}")
    print(f"Loss: {metrics['loss']:.4f}")

    # Mostra onde os arquivos foram salvos.
    print(f"Modelo: {MODEL_FILE}")
    print(f"Histórico: {HISTORY_FILE}")
    print(f"Métricas: {METRICS_FILE}")
    print(f"Pasta dos gráficos: {FIGURES_DIR}")

    # Lista os gráficos disponíveis para abrir no VS Code.
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {figure.name}")


def main():
    # Define as opções disponíveis no terminal.
    parser = argparse.ArgumentParser(
        description="Classificador CIFAR-10"
    )

    parser.add_argument(
        "--etapa",
        choices=("tudo", "dados", "treino", "resultados"),
        default="tudo",
        help=(
            "Escolhe quais etapas executar. "
            "O padrão executa todo o fluxo."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    args = parser.parse_args()

    # Valida o número de épocas quando houver treinamento.
    if args.etapa in ("tudo", "treino") and args.epochs < 1:
        parser.error("--epochs deve ser maior que zero.")

    # Carrega os dados, normaliza e gera a figura com os exemplos.
    if args.etapa in ("tudo", "dados"):
        print("\nPreparando os dados...")

        dataset.main()
        features.main()

    # Treina uma rede nova e salva o modelo e o histórico.
    # Essa etapa substitui o modelo anteriormente salvo.
    if args.etapa in ("tudo", "treino"):
        print("\nTreinando o modelo...")

        train.main(epochs=args.epochs)

    # Avalia o modelo salvo e gera a figura das previsões.
    # A opção resultados executa esta parte sem treinar novamente.
    if args.etapa in ("tudo", "treino", "resultados"):
        print("\nAvaliando o modelo...")

        predict.main()
        show_results(),
    # Abre as imagens no visualizador padrão do Windows.
    # Os arquivos continuam salvos em reports/figures.
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        os.startfile(str(figure))


# Inicia a execução ao chamar python main.py.
if __name__ == "__main__":
    main()


    # MODOS DE EXECUÇÃO
# Execute os comandos na raiz do projeto, onde está o pyproject.toml.
#
# 1. FLUXO COMPLETO — modo padrão
# Prepara os dados, treina, avalia e gera os gráficos e resultados.
# uv run python -m module_classificador_imagens.main
#
# Também pode ser informado explicitamente:
# uv run python -m module_classificador_imagens.main --etapa tudo
#
# 2. SOMENTE DADOS
# Carrega a CIFAR-10, normaliza as imagens, prepara os rótulos
# e gera o gráfico com exemplos da base.
# uv run python -m module_classificador_imagens.main --etapa dados
#
# 3. TREINAMENTO E AVALIAÇÃO
# Utiliza os dados já preparados, treina um modelo novo,
# avalia e salva os gráficos e resultados.
# Requer que a preparação dos dados tenha sido executada.
# uv run python -m module_classificador_imagens.main --etapa treino
#
# 4. SOMENTE RESULTADOS
# Carrega o modelo salvo, avalia o conjunto de teste, gera as
# previsões e recria as curvas a partir do histórico disponível.
# Não treina novamente. Requer dados preparados e modelo salvo.
# uv run python -m module_classificador_imagens.main --etapa resultados
#
# 5. FLUXO COMPLETO COM NÚMERO PERSONALIZADO DE ÉPOCAS
# Exemplo: executar todo o fluxo com no máximo 10 épocas.
# uv run python -m module_classificador_imagens.main --epochs 10
#
# 6. TREINAMENTO COM NÚMERO PERSONALIZADO DE ÉPOCAS
# Exemplo: treinar e avaliar com no máximo 20 épocas.
# uv run python -m module_classificador_imagens.main --etapa treino --epochs 20
#
# 7. VERIFICAÇÃO COM UMA ÉPOCA
# Executa todo o fluxo com uma época de treinamento.
# uv run python -m module_classificador_imagens.main --epochs 1
#
# 8. AJUDA
# Exibe as opções disponíveis no terminal.
# uv run python -m module_classificador_imagens.main --help
#
# OBSERVAÇÕES
# --epochs deve ser um inteiro positivo nos modos com treinamento.
# Sem --epochs, o limite é EPOCHS definido em config.py.
# O EarlyStopping pode encerrar o treinamento antes desse limite.
# --epochs não é utilizado nos modos dados e resultados.
# Cada treinamento começa do zero e substitui o modelo anterior.
# Os gráficos são salvos em reports/figures.
# O histórico e as métricas são salvos em reports.
# O modelo treinado é salvo em models.