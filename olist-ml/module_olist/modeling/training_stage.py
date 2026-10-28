"""Etapa separada de treinamento final do modelo.

Este módulo explicita a parte da pipeline dedicada ao treinamento do modelo
final após a seleção do melhor candidato pela validação cruzada.
"""

from __future__ import annotations

from pathlib import Path

from module_olist.modeling.train import train_model


def train_final_model(model_name: str, threshold: float, X_train, y_train, model_path: Path, metadata_path: Path):
    """Wrapper explícito para a etapa de treinamento final do modelo."""
    return train_model(
        model_name=model_name,
        threshold=threshold,
        X_train=X_train,
        y_train=y_train,
        model_path=model_path,
        metadata_path=metadata_path,
    )
