# credit_scoring.py
# Clean, ready-to-run script for German Credit Scoring
# Run this file using: python credit_scoring.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------
# STEP 1 — Locate the dataset
# --------------------------------------------------------

def find_csv():
    """
    Try to find GermanCredit.csv in project/data/ or fallback path.
    """
    project_root = Path(".")
    data_local = project_root / "data" / "GermanCredit.csv"
    fallback = Path("/mnt/data/GermanCredit.csv")

    if data_local.exists():
        return data_local
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"GermanCredit.csv not found.\n"
        f"Expected at: {data_local} or {fallback}"
    )

# --------------------------------------------------------
# STEP 2 — Load dataset
# --------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df

# --------------------------------------------------------
# STEP 3 — Detect target column
# --------------------------------------------------------

def detect_target(df):
    possible = ["credit_risk", "Creditability", "creditability", "Risk", "default"]

    for col in possible:
        if col in df.columns:
            return col

    # fallback: last column if binary
    last_col = df.columns[-1]
    uniq = set(df[last_col].dropna().unique())
    if uniq.issubset({0, 1, "0", "1"}):
        return last_col

    raise ValueError("Cannot detect target column!")

# --------------------------------------------------------
# STEP 4 — Build preprocessor
# --------------------------------------------------------

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Handle OneHotEncoder across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    categorical_transformer = Pipeline(steps=[("onehot", ohe)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )
    return preprocessor

# --------------------------------------------------------
# STEP 5 — Train & Evaluate Models
# --------------------------------------------------------

def train_models(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "logreg": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000))
        ])
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier

        models["xgb"] = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    except Exception:
        print("XGBoost not installed. Only Logistic Regression will run.")

    results = {}

    # Train each model
    for name, model in models.items():
        print(f"\n=== Training {name.upper()} ===")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        try:
            probs = model.predict_proba(X_test)[:, 1]
        except Exception:
            probs = None

        report = classification_report(y_test, preds)
        print(report)

        auc = roc_auc_score(y_test, probs) if probs is not None else 0
        print("ROC AUC:", auc)

        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:\n", cm)

        results[name] = {"model": model, "auc": auc}

    # choose best
    best = max(results, key=lambda m: results[m]["auc"])
    print(f"\n🔥 BEST MODEL: {best}  (AUC = {results[best]['auc']})")

    return results[best]["model"], X_test, y_test

# --------------------------------------------------------
# STEP 6 — Save results
# --------------------------------------------------------

def save_outputs(model, X_test, y_test):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    joblib.dump(model, model_path)
    print("Saved model →", model_path)

    preds = model.predict(X_test)

    try:
        probs = model.predict_proba(X_test)[:, 1]
    except:
        probs = np.nan

    out = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "probability": probs
    })

    out_csv = models_dir / "test_predictions.csv"
    out.to_csv(out_csv, index=False)
    print("Saved predictions →", out_csv)

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():
    csv_path = find_csv()
    print("Dataset loaded from:", csv_path)

    df = load_data(csv_path)

    print("\n=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head().to_string())

    target_col = detect_target(df)
    print("\nDetected target column:", target_col)

    # convert string labels if needed
    df[target_col] = df[target_col].replace({"good": 0, "bad": 1, "yes": 1, "no": 0})

    # train
    model, X_test, y_test = train_models(df, target_col)

    # save outputs
    save_outputs(model, X_test, y_test)

    print("\n🎉 ALL DONE!")

if __name__ == "__main__":
    main()
