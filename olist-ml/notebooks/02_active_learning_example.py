"""Exemplo de active learning com Keras para a etapa da Aula 12.

Este exemplo demonstra uma estratégia clássica de active learning:
1. treina um modelo inicial em um conjunto pequeno de dados rotulados;
2. calcula a incerteza das amostras do pool não rotulado;
3. seleciona as amostras mais incertas para revisão ou anotação;
4. usa o Keras para treinar o modelo base.

Esse padrão é compatível com a ideia de active learning e evidencia que o
conceito é independente da biblioteca, mas nesse caso foi implementado com Keras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense


def build_model(input_dim: int = 2) -> Sequential:
    """Cria uma rede simples para classificação binária."""
    model = Sequential(
        [
            Dense(8, activation="relu", input_shape=(input_dim,)),
            Dense(4, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Calcula entropia para medir incerteza das previsões."""
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return -(probabilities * np.log(probabilities) + (1.0 - probabilities) * np.log(1.0 - probabilities))


def select_uncertain_samples(model: Sequential, X_pool: np.ndarray, n_queries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Seleciona as amostras mais incertas do conjunto não rotulado."""
    probabilities = model.predict(X_pool, verbose=0).ravel()
    uncertainty = compute_entropy(probabilities)
    top_indices = np.argsort(uncertainty)[::-1][:n_queries]
    return top_indices, uncertainty[top_indices]


def main() -> None:
    rng = np.random.default_rng(42)

    X_labeled = np.array(
        [
            [-1.0, -1.2],
            [-0.9, -0.6],
            [-1.1, 0.3],
            [1.0, 0.8],
            [0.9, 1.3],
            [1.1, -0.2],
        ],
        dtype=float,
    )
    y_labeled = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    X_pool = rng.normal(loc=0.0, scale=1.3, size=(12, 2)).astype(float)

    model = build_model(input_dim=X_labeled.shape[1])
    model.fit(X_labeled, y_labeled, epochs=200, verbose=0)

    selected_idx, selected_uncertainty = select_uncertain_samples(model, X_pool, n_queries=3)

    pool_df = pd.DataFrame(X_pool, columns=["feature_1", "feature_2"])
    pool_df["uncertainty"] = compute_entropy(model.predict(X_pool, verbose=0).ravel())

    print("Amostras candidatas para rotulagem ativa:")
    print(pool_df.iloc[selected_idx].sort_values("uncertainty", ascending=False).round(4))
    print(f"Total selecionado: {len(selected_idx)}")
    print("Observação: as amostras mais incertas são as melhores candidatas para revisão em active learning.")


if __name__ == "__main__":
    main()
