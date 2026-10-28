"""Exemplo de active learning para a etapa da Aula 12.

Este script demonstra uma estratégia segura e didática de seleção por incerteza:
um modelo inicial é treinado em um conjunto rotulado pequeno e, em seguida,
seleciona as amostras mais incertas no pool não rotulado para futura revisão.

A implementação evita dependência de Keras e foca no conceito de active learning,
que é independente do framework usado para treinar o modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def compute_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Calcula a incerteza por entropia para probabilidades binárias."""
    probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    entropy = -(probabilities * np.log(probabilities) + (1.0 - probabilities) * np.log(1.0 - probabilities))
    return entropy


def select_uncertain_samples(model: LogisticRegression, X_pool: np.ndarray, n_queries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Seleciona as amostras mais incertas do conjunto não rotulado."""
    probabilities = model.predict_proba(X_pool)
    uncertainty = compute_uncertainty(probabilities[:, 1])
    top_indices = np.argsort(uncertainty)[::-1][:n_queries]
    return top_indices, uncertainty[top_indices]


def main() -> None:
    rng = np.random.default_rng(42)

    X_labeled = np.array(
        [
            [-1.0, -1.0],
            [-0.8, -0.6],
            [-0.9, 0.4],
            [1.0, 0.8],
            [0.8, 1.0],
            [0.7, -0.2],
        ],
        dtype=float,
    )
    y_labeled = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    X_pool = rng.normal(loc=0.0, scale=1.3, size=(12, 2))

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_labeled, y_labeled)

    selected_idx, selected_uncertainty = select_uncertain_samples(model, X_pool, n_queries=3)

    pool_df = pd.DataFrame(X_pool, columns=["feature_1", "feature_2"])
    pool_df["uncertainty"] = compute_uncertainty(model.predict_proba(X_pool)[:, 1])

    print("Amostras candidatas para rotulagem ativa:")
    print(pool_df.iloc[selected_idx].sort_values("uncertainty", ascending=False).round(4))
    print(f"Total selecionado: {len(selected_idx)}")


if __name__ == "__main__":
    main()
