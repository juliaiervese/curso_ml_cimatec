"""Exemplo mínimo de active learning para a etapa da Aula 12.

A ideia é demonstrar a seleção de amostras mais informativas para retraining,
sem necessidade de um pipeline completo de produção.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def select_uncertain_samples(probabilities: np.ndarray, uncertainty_threshold: float = 0.7) -> np.ndarray:
    """Retorna índices das amostras mais incertas."""
    uncertainty = np.abs(probabilities - 0.5)
    return np.where(uncertainty >= uncertainty_threshold)[0]


def main() -> None:
    sample = pd.DataFrame(
        {
            "score": [0.12, 0.33, 0.51, 0.77, 0.88, 0.46],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )

    uncertain_idx = select_uncertain_samples(sample["score"].to_numpy())
    print("Amostras candidatas para anotação ativa:")
    print(sample.iloc[uncertain_idx])
    print(f"Total selecionado: {len(uncertain_idx)}")


if __name__ == "__main__":
    main()
