"""Etapa separada de explicabilidade com SHAP.

Esta etapa é independente da inferência e evidencia a análise de importância das
features do modelo em amostras selecionadas.
"""

from __future__ import annotations

from module_olist.explain import generate_explanations


def generate_model_explanations(data=None, model=None, model_name=None, sample_size=100):
    """Wrapper explícito para a etapa de explicabilidade SHAP."""
    return generate_explanations(
        data=data,
        model=model,
        model_name=model_name,
        sample_size=sample_size,
    )
