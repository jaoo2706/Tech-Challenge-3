"""Extrai as tabelas da camada Gold (BigQuery) para arquivos Parquet locais.

Uso:
    python -m src.preprocessing.extract_gold_data

Requer autenticação prévia via `gcloud auth application-default login`
com permissão de leitura no projeto/dataset configurados abaixo.
"""
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "tech-challenge-2-fiap-joao"
DATASET = "gold_alfabetizacao"
TABLES = [
    "indicador_alfabetizacao_municipio",
    "metas_vs_resultados_municipio",
    "metas_vs_resultados_uf",
    "metas_vs_resultados_brasil",
    "evolucao_temporal_uf",
    "evolucao_temporal_regiao",
    "evolucao_temporal_brasil",
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def extract() -> None:
    client = bigquery.Client(project=PROJECT_ID)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for table in TABLES:
        query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{table}`"
        df = client.query(query).to_dataframe()
        out_path = RAW_DIR / f"{table}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[ok] {table}: {len(df)} linhas -> {out_path.relative_to(RAW_DIR.parents[1])}")


if __name__ == "__main__":
    extract()
