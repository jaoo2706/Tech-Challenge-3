"""Gera tabelas de apoio às perguntas de negócio do desafio:
risco por município, padrões regionais e ranking de risco futuro.

Uso:
    python -m src.evaluation.business_insights
"""
from pathlib import Path

import joblib
import pandas as pd

from src.modeling.train import DROP_COLS, TARGET, temporal_split

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "base_analitica.parquet")
    df_model = df.drop(columns=["nome_uf"])
    _, test_df = temporal_split(df_model)
    X_test = test_df.drop(columns=DROP_COLS)

    pipeline = joblib.load(ROOT / "models" / "modelo_alfabetizacao.joblib")
    proba_atingir_meta = pipeline.predict_proba(X_test)[:, 1]

    ranking = test_df[["id_municipio", "nome_municipio", "sigla_uf"]].copy()
    ranking["probabilidade_atingir_meta_2024"] = proba_atingir_meta
    ranking["classificacao_risco"] = pd.cut(
        ranking["probabilidade_atingir_meta_2024"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Alto risco", "Risco moderado", "Baixo risco"],
    )
    ranking = ranking.sort_values("probabilidade_atingir_meta_2024")
    ranking.to_csv(REPORTS_DIR / "ranking_risco_municipios_2024.csv", index=False)

    # Padrões regionais (a partir dos dados reais, não do modelo)
    resumo_regiao = (
        df.groupby("regiao")
        .agg(
            municipios=("id_municipio", "nunique"),
            taxa_atingimento_meta=("alfabetizacao_adequada", "mean"),
        )
        .sort_values("taxa_atingimento_meta")
    )
    resumo_regiao.to_csv(REPORTS_DIR / "resumo_taxa_atingimento_regiao.csv")

    resumo_uf = (
        df.groupby(["regiao", "sigla_uf"])
        .agg(taxa_atingimento_meta=("alfabetizacao_adequada", "mean"))
        .sort_values("taxa_atingimento_meta")
    )
    resumo_uf.to_csv(REPORTS_DIR / "resumo_taxa_atingimento_uf.csv")

    print("=== Top 15 municípios de MAIOR risco (menor probabilidade de atingir a meta em 2024) ===")
    print(ranking.head(15).to_string(index=False))
    print()
    print("=== Taxa de atingimento da meta por região (2022-2024) ===")
    print(resumo_regiao)
    print()
    print("=== 5 UFs com menor taxa de atingimento ===")
    print(resumo_uf.head(5))
    print()
    print("=== 5 UFs com maior taxa de atingimento ===")
    print(resumo_uf.tail(5))
    print(f"\n[ok] Tabelas salvas em {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
