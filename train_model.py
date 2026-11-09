"""
Student Performance Prediction System
Downloads the UCI Mathematics dataset, trains Random Forest and Logistic Regression
on the SAME stratified train/test split, evaluates both, and repeats without G1/G2.
Run: python train_model.py
"""
from pathlib import Path
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.20

def get_data():
    csv_path = ROOT / "student_data.csv"
    if csv_path.exists() and csv_path.stat().st_size > 100:
        df = pd.read_csv(csv_path)
        if "G3" in df.columns and len(df) > 10:
            return df
    print("Downloading UCI Student Performance Mathematics file...")
    # Prefer the official UCI ZIP and explicitly select student-mat.csv.
    import urllib.request, zipfile, io
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(request, timeout=45).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        member = next(n for n in zf.namelist() if n.lower().endswith("student-mat.csv"))
        df = pd.read_csv(zf.open(member), sep=";")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the official UCI Mathematics CSV. "
            "Check internet access and rerun. Download student-mat.csv from "
            "https://archive.ics.uci.edu/dataset/320/student+performance and place it "
            "beside train_model.py, or retry. Details: " + str(exc)
        )
    if "G3" not in df.columns or len(df) != 395:
        raise ValueError(f"Expected the 395-row Mathematics file; received shape {df.shape}.")
    df.to_csv(csv_path, index=False)
    return df

def build_pipeline(X, estimator):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
    return Pipeline([("preprocessor", prep), ("classifier", estimator)])

def metrics_row(name, y_test, pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0)
    }

def evaluate_experiment(df, omit_grades=False, suffix="full"):
    clean = df.drop_duplicates().copy()
    y = (clean["G3"] < 10).astype(int)
    drop_cols = ["G3"] + (["G1", "G2"] if omit_grades else [])
    X = clean.drop(columns=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        )
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = build_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(metrics_row(name, y_test, pred))
        fitted[name] = (pipe, pred)
        print(f"\n{name} — {suffix}")
        print(classification_report(y_test, pred, target_names=["Not at risk", "At risk"], zero_division=0))
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test, pred), display_labels=["Not at risk", "At risk"]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} — Confusion Matrix ({suffix})")
        plt.tight_layout()
        plt.savefig(ROOT / f"confusion_matrix_{name.lower().replace(' ', '_')}_{suffix}.png", dpi=160)
        plt.close()

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / f"evaluation_{suffix}.csv", index=False)
    return X_train, X_test, y_train, y_test, fitted, results

def main():
    df = get_data()
    if "G3" not in df.columns:
        raise ValueError("Dataset must contain G3 final Mathematics grade.")
    print("Dataset shape:", df.shape)
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Missing values:", int(df.isna().sum().sum()))
    print("At-risk counts:", (df["G3"] < 10).value_counts().to_dict())
    # Full experiment: G1 and G2 included; G3 never used as a feature.
    Xtr, Xte, ytr, yte, fitted, full_results = evaluate_experiment(df, False, "full")
    # Save the Random Forest model for the Streamlit app.
    rf = fitted["Random Forest"][0]
    joblib.dump(rf, ROOT / "student_model.pkl")
    # Permutation importance is model-agnostic and computed on held-out data.
    perm = permutation_importance(rf, Xte, yte, n_repeats=15,
                                  random_state=RANDOM_STATE, scoring="f1")
    imp = pd.DataFrame({"Feature": Xte.columns, "Importance": perm.importances_mean})
    imp = imp.sort_values("Importance", ascending=False)
    imp.to_csv(ROOT / "feature_importance.csv", index=False)
    plt.figure(figsize=(9, 6))
    top = imp.head(15).sort_values("Importance")
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Mean decrease in F1 after permutation")
    plt.title("Random Forest Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(ROOT / "feature_importance.png", dpi=160)
    plt.close()
    # Early-cycle experiment: remove both G1 and G2.
    _, _, _, _, _, early_results = evaluate_experiment(df, True, "early_no_G1_G2")
    combined = pd.concat([
        full_results.assign(Experiment="With G1 and G2"),
        early_results.assign(Experiment="Without G1 and G2")
    ], ignore_index=True)
    combined.to_csv(ROOT / "model_comparison_all.csv", index=False)
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicates_removed": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "target_rule": "G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }
    (ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved: student_model.pkl, evaluation CSVs, confusion matrices, feature importance.")
    print(combined.to_string(index=False))

if __name__ == "__main__":
    main()
