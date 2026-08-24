"""Constrói a base analítica (features + alvo) a partir da camada Gold.

A camada Gold está agregada em nível município-ano (não há dado por aluno).
O alvo é reformulado como: o município atingiu ou não a meta de alfabetização
no ano (`alfabetizacao_adequada`), derivado de `gap_meta_percentual >= 0`.

Tratamento de data leakage
---------------------------
`indicador_alfabetizacao_percentual`, `gap_meta_percentual` e
`proficiencia_media_saeb` do MESMO ano são o resultado (ou derivam
diretamente dele) e são excluídos das features do ano corrente. Em vez
disso, usamos versões defasadas (ano anterior, por município) dessas
variáveis como features -- capturam a trajetória histórica do município
sem revelar o resultado que queremos prever.

Uso:
    python -m src.preprocessing.build_features
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Colunas do ano corrente que definem/vazam o alvo -> nunca usar como feature.
LEAKY_COLUMNS = [
    "indicador_alfabetizacao_percentual",
    "gap_meta_percentual",
    "proficiencia_media_saeb",
]

# Colunas defasadas (histórico do município) que serão criadas.
LAG_SOURCE_COLUMNS = [
    "indicador_alfabetizacao_percentual",
    "proficiencia_media_saeb",
    "gap_meta_percentual",
    "meta_percentual",
    "total_estudantes_avaliados",
    "total_escolas_avaliadas",
    "alfabetizacao_adequada",
]


def load_base() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "indicador_alfabetizacao_municipio.parquet")
    df = df.sort_values(["id_municipio", "ano"]).reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["alfabetizacao_adequada"] = (df["gap_meta_percentual"] >= 0).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lagged = df[["id_municipio", "ano"] + LAG_SOURCE_COLUMNS].copy()
    lagged["ano"] = lagged["ano"] + 1  # o valor do ano N vira feature do ano N+1
    rename_map = {c: f"{c}_lag1" for c in LAG_SOURCE_COLUMNS}
    lagged = lagged.rename(columns=rename_map)

    out = df.merge(lagged, on=["id_municipio", "ano"], how="left")

    # tendência: variação do indicador entre ano-2 e ano-1 (2 anos de histórico)
    out["indicador_tendencia_lag1"] = out.groupby("id_municipio")[
        "indicador_alfabetizacao_percentual_lag1"
    ].diff()

    return out


def build(min_ano_treino: int = 2022) -> pd.DataFrame:
    df = load_base()
    df = add_target(df)
    df = add_lag_features(df)

    # Remove anos sem histórico suficiente (2021 não tem lag) e o ano
    # incompleto (2025, com apenas 1 registro na Gold).
    df = df[(df["ano"] >= min_ano_treino) & (df["ano"] <= 2024)].reset_index(drop=True)

    # Remove as colunas do ano corrente que vazam o alvo.
    df = df.drop(columns=LEAKY_COLUMNS)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "base_analitica.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[ok] base analítica: {len(df)} linhas, {df.shape[1]} colunas -> {out_path}")
    print(f"     anos: {sorted(df['ano'].unique())}")
    print(f"     balanceamento do alvo:\n{df['alfabetizacao_adequada'].value_counts(normalize=True)}")
    return df


if __name__ == "__main__":
    build()
