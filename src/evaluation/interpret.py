"""Gera artefatos de avaliação e interpretabilidade do modelo final:
matriz de confusão, curva ROC, feature importance e SHAP values.

Uso:
    python -m src.evaluation.interpret
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.modeling.train import DROP_COLS, TARGET, temporal_split

ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def get_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "base_analitica.parquet")
    df = df.drop(columns=["nome_uf"])
    _, test_df = temporal_split(df)
    X_test = test_df.drop(columns=DROP_COLS)
    y_test = test_df[TARGET]

    pipeline = joblib.load(MODELS_DIR / "modelo_alfabetizacao.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- Matriz de confusão ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Não atingiu meta", "Atingiu meta"], ax=ax, cmap="Blues"
    )
    ax.set_title("Matriz de Confusão — Holdout 2024")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # --- Curva ROC ---
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Curva ROC — Holdout 2024")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    # --- Feature importance (coeficientes ou impurity-based) ---
    feature_names = get_feature_names(preprocessor)
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    if hasattr(model, "coef_"):
        importances = model.coef_[0]
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df["abs_importance"] = imp_df["importance"].abs()
        imp_df = imp_df.sort_values("abs_importance", ascending=False).head(15)
        title = "Feature Importance (coeficientes, Regressão Logística)"
    else:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        title = "Feature Importance (impurity-based)"

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in imp_df["importance"]]
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    imp_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # --- SHAP values ---
    background = shap.sample(X_test_transformed, min(100, len(X_test_transformed)), random_state=42)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X_test_transformed, feature_names=feature_names, show=False, max_display=15
    )
    plt.title("SHAP Summary — impacto das variáveis na predição")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("[ok] Artefatos gerados em images/ e reports/:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png / feature_importance.csv")
    print("  - shap_summary.png")
    print("\nTop 10 features mais importantes:")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
