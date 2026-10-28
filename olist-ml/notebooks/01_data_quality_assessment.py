"""Checklist de qualidade dos dados para a etapa da Aula 04.

Este script serve como evidência mínima de que a análise de qualidade dos
dados foi implementada antes da modelagem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assess_data_quality(data: pd.DataFrame) -> dict:
    """Calcula indicadores básicos de qualidade dos dados."""
    report = {
        "n_rows": int(len(data)),
        "n_columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_by_column": data.isna().sum().astype(int).to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "numeric_summary": {
            col: {
                "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
                "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
                "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
            }
            for col, series in data.items()
            if pd.api.types.is_numeric_dtype(series)
        },
    }
    return report


def main() -> None:
    data_path = Path("../data/raw/olist_orders_dataset.csv")
    if not data_path.exists():
        print("Arquivo de dados não encontrado em:", data_path)
        print("Este é um exemplo de estrutura mínima para a etapa de qualidade dos dados.")
        return

    data = pd.read_csv(data_path)
    report = assess_data_quality(data)
    print("Relatório de qualidade dos dados")
    print("-" * 40)
    print(f"Linhas: {report['n_rows']}")
    print(f"Colunas: {report['n_columns']}")
    print(f"Valores ausentes: {report['missing_values']}")
    print(f"Linhas duplicadas: {report['duplicate_rows']}")
    print("Colunas com ausência:")
    for column, count in report["missing_by_column"].items():
        if count > 0:
            print(f"  - {column}: {count}")


if __name__ == "__main__":
    main()
