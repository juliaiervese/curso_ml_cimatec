"""Etapa separada de seleção do melhor modelo via validação cruzada.

Este módulo evidencia que a escolha do modelo e do threshold é uma etapa distinta
antes do treinamento final do pipeline.
"""

from __future__ import annotations

from module_olist.modeling.cross_validation import cross_validate_models


def select_best_model_with_cv(X_train, y_train):
    """Runs the validation-only stage for model selection."""
    return cross_validate_models(X_train, y_train)
