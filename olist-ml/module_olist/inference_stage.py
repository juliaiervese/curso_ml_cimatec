"""Etapa separada de inferência do modelo.

A inferência em dados novos representa uma fase distinta do treinamento e da
explicabilidade do modelo.
"""

from __future__ import annotations

from module_olist.inference import run_inference


def execute_inference(data=None, model=None, model_name=None, threshold=None, sample_size=5, output_path=None):
    """Wrapper explícito para a etapa de inferência do pipeline."""
    return run_inference(
        data=data,
        model=model,
        model_name=model_name,
        threshold=threshold,
        sample_size=sample_size,
        output_path=output_path,
    )
