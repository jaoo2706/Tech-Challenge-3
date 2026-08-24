"""Treina e valida modelos supervisionados para prever se um município
atingirá a meta de alfabetização (`alfabetizacao_adequada`).

Pipeline (100% integrada via sklearn.Pipeline -> sem leakage no pré-processamento):
  - Imputação de numéricas (mediana) e categóricas (moda)
  - Encoding one-hot para categóricas (regiao, sigla_uf)
  - Padronização de numéricas
  - Classificador (comparação entre LogisticRegression, RandomForest, GradientBoosting)

Validação:
  - Split temporal: treino = 2022-2023, teste = 2024 (simula previsão do futuro)
  - GroupKFold (por município) dentro do treino para tuning -> evita que o
    mesmo município apareça em treino e validação em anos diferentes
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET = "alfabetizacao_adequada"
ID_COLS = ["id_municipio", "nome_municipio"]
DROP_COLS = ID_COLS + [TARGET]

NUMERIC_FEATURES = [
    "total_estudantes_avaliados",
    "total_escolas_avaliadas",
    "meta_percentual",
    "indicador_alfabetizacao_percentual_lag1",
    "proficiencia_media_saeb_lag1",
    "gap_meta_percentual_lag1",
    "meta_percentual_lag1",
    "total_estudantes_avaliados_lag1",
    "total_escolas_avaliadas_lag1",
    "alfabetizacao_adequada_lag1",
    "indicador_tendencia_lag1",
]
# `regiao` é omitida das features: é uma agregação direta/determinística de
# `sigla_uf` (nested), então incluir as duas gera colinearidade perfeita
# entre grupos de dummies, o que infla/instabiliza os coeficientes da
# regressão logística e prejudica a interpretabilidade (SHAP/feature
# importance). Mantemos apenas `sigla_uf`, mais granular; leituras
# regionais são obtidas agregando as UFs em análise posterior (ver EDA).
CATEGORICAL_FEATURES = ["sigla_uf"]
# `ano` entra como feature ordinal simples (tendência temporal nacional)
ORDINAL_FEATURES = ["ano"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ordinal_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("ord", ordinal_pipe, ORDINAL_FEATURES),
        ]
    )


def get_candidate_models() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
        ),
    }


def temporal_split(df: pd.DataFrame):
    train = df[df["ano"] <= 2023].reset_index(drop=True)
    test = df[df["ano"] == 2024].reset_index(drop=True)
    return train, test


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "base_analitica.parquet")
    df = df.drop(columns=["nome_uf"])  # redundante com sigla_uf/regiao, alta cardinalidade

    train_df, test_df = temporal_split(df)

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[TARGET]
    groups_train = train_df["id_municipio"]

    X_test = test_df.drop(columns=DROP_COLS)
    y_test = test_df[TARGET]

    print(f"Treino: {X_train.shape}, Teste (holdout temporal 2024): {X_test.shape}")
    print(f"Balanceamento treino:\n{y_train.value_counts(normalize=True)}\n")

    cv = GroupKFold(n_splits=5)
    results = {}

    for name, (estimator, param_grid) in get_candidate_models().items():
        pipe = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        cv_splits = list(cv.split(X_train, y_train, groups=groups_train))
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=8,
            scoring="roc_auc",
            cv=cv_splits,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": search.best_score_,
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_f1": f1_score(y_test, y_pred),
            "pipeline": best_pipe,
        }

        print(f"=== {name} ===")
        print(f"  Melhor CV ROC-AUC (GroupKFold, treino): {search.best_score_:.4f}")
        print(f"  ROC-AUC (holdout 2024): {results[name]['test_roc_auc']:.4f}")
        print(f"  F1 (holdout 2024): {results[name]['test_f1']:.4f}")
        print(f"  Melhores hiperparâmetros: {search.best_params_}")
        print(classification_report(y_test, y_pred, digits=3))
        print()

    best_name = max(results, key=lambda k: results[k]["test_roc_auc"])
    best_pipeline = results[best_name]["pipeline"]
    print(f">>> Modelo selecionado: {best_name} (ROC-AUC holdout = {results[best_name]['test_roc_auc']:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "modelo_alfabetizacao.joblib")
    joblib.dump(
        {"feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES},
        MODELS_DIR / "feature_metadata.joblib",
    )

    summary = pd.DataFrame(
        {
            name: {
                "cv_roc_auc": r["cv_roc_auc"],
                "test_roc_auc": r["test_roc_auc"],
                "test_f1": r["test_f1"],
            }
            for name, r in results.items()
        }
    ).T
    summary.to_csv(Path(__file__).resolve().parents[2] / "reports" / "model_comparison.csv")
    print(f"\nResumo salvo em reports/model_comparison.csv:\n{summary}")


if __name__ == "__main__":
    main()
