import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.caption("Academic-risk exploration using the UCI Student Performance Mathematics dataset.")
st.warning("Classroom prototype only. Predictions are not diagnoses and must not be the sole basis for real student decisions.")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def run_experiments(data):
    df = data.drop_duplicates().copy()
    y = (df["G3"] < 10).astype(int)
    X_full = df.drop(columns=["G3"])
    X_early = X_full.drop(columns=["G1", "G2"], errors="ignore")

    # Identical split settings and stratification for both feature sets.
    split_full = train_test_split(
        X_full, y, test_size=0.20, random_state=42, stratify=y
    )
    split_early = train_test_split(
        X_early, y, test_size=0.20, random_state=42, stratify=y
    )

    def make_preprocessor(X):
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
        return ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical)
        ])

    experiments = {}
    for label, X, split in [
        ("Full features (with G1 and G2)", X_full, split_full),
        ("Early prediction (without G1 and G2)", X_early, split_early)
    ]:
        X_train, X_test, y_train, y_test = split
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            )
        }
        for model_name, classifier in models.items():
            pipe = Pipeline([
                ("preprocessor", make_preprocessor(X)),
                ("classifier", classifier)
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            experiments[(label, model_name)] = {
                "model": pipe, "X_test": X_test, "y_test": y_test,
                "pred": pred, "X_columns": X.columns.tolist(),
                "metrics": {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (At risk)": precision_score(y_test, pred, zero_division=0),
                    "Recall (At risk)": recall_score(y_test, pred, zero_division=0),
                    "F1-score (At risk)": f1_score(y_test, pred, zero_division=0)
                }
            }
    return experiments, df, y

try:
    data = load_data()
except FileNotFoundError:
    st.error("student_data.csv was not found. Run `python train_model.py` first, then place the CSV in this project folder.")
    st.stop()

if "G3" not in data.columns:
    st.error("The dataset must contain the G3 final-grade column.")
    st.stop()

with st.spinner("Training and evaluating four model experiments..."):
    experiments, clean_data, target = run_experiments(data)

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Student records", f"{len(clean_data):,}")
c2.metric("At-risk records", int(target.sum()))
c3.metric("Not-at-risk records", int((target == 0).sum()))
st.dataframe(clean_data.head(10), use_container_width=True)

st.subheader("1. Model comparison")
rows = []
for (feature_set, model_name), result in experiments.items():
    rows.append({"Experiment": feature_set, "Model": model_name, **result["metrics"]})
comparison = pd.DataFrame(rows)
st.dataframe(comparison.style.format({
    "Accuracy": "{:.3f}", "Precision (At risk)": "{:.3f}",
    "Recall (At risk)": "{:.3f}", "F1-score (At risk)": "{:.3f}"
}), use_container_width=True)
st.bar_chart(comparison.set_index(["Experiment", "Model"])[["Accuracy", "Precision (At risk)", "Recall (At risk)", "F1-score (At risk)"]])

st.subheader("2. Confusion matrices")
st.write("Rows represent actual classes; columns represent predicted classes. Class 1 means AT RISK.")
choice = st.selectbox("Choose experiment", list(experiments.keys()), format_func=lambda x: f"{x[0]} — {x[1]}")
selected = experiments[choice]
cm = confusion_matrix(selected["y_test"], selected["pred"], labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Not at risk", "At risk"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d"
)
ax.set_title(f"{choice[1]} — {choice[0]}")
st.pyplot(fig)
plt.close(fig)
tn, fp, fn, tp = cm.ravel()
st.write(f"**TN:** {tn} · **FP:** {fp} · **FN:** {fn} · **TP:** {tp}")
st.code(classification_report(
    selected["y_test"], selected["pred"],
    target_names=["Not at risk", "At risk"], zero_division=0
))

st.subheader("3. Feature importance and interpretation")
st.write("Random Forest importance is based on impurity reduction. Logistic Regression coefficients show the direction and strength of the fitted linear relationship; absolute coefficient size is used for ranking.")
importance_choice = st.selectbox(
    "Feature-importance experiment",
    ["Full features (with G1 and G2)", "Early prediction (without G1 and G2)"]
)
rf = experiments[(importance_choice, "Random Forest")]["model"]
lr = experiments[(importance_choice, "Logistic Regression")]["model"]
feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
rf_values = rf.named_steps["classifier"].feature_importances_
lr_values = lr.named_steps["classifier"].coef_[0]
imp = pd.DataFrame({
    "Feature": feature_names,
    "Random Forest importance": rf_values,
    "Logistic Regression coefficient": lr_values,
    "Absolute LR coefficient": np.abs(lr_values)
})
top_rf = imp.sort_values("Random Forest importance", ascending=False).head(15)
st.markdown("**Top 15 Random Forest features**")
st.bar_chart(top_rf.set_index("Feature")["Random Forest importance"])
top_lr = imp.sort_values("Absolute LR coefficient", ascending=False).head(15)
st.markdown("**Top 15 Logistic Regression coefficients (ranked by absolute value)**")
st.bar_chart(top_lr.set_index("Feature")["Logistic Regression coefficient"])
st.caption("One-hot encoded categories appear as separate feature names. Importance indicates model association, not causation.")

st.subheader("4. Full-information vs. early academic prediction")
wide = comparison.pivot(index="Model", columns="Experiment", values=["Accuracy", "Recall (At risk)", "F1-score (At risk)"])
st.dataframe(wide.style.format("{:.3f}"), use_container_width=True)
st.info("The early-prediction experiment removes G1 and G2 from the input data but keeps the same target (G3 < 10) and the same 80/20 stratified split settings. Compare recall and F1 for the At risk class, not accuracy alone.")

st.subheader("5. Predict uploaded student records")
mode = st.radio("Prediction timing", ["Full features (with G1 and G2)", "Early prediction (without G1 and G2)"], horizontal=True)
model_for_upload = experiments[(mode, "Random Forest")]["model"]
required = experiments[(mode, "Random Forest")]["X_columns"]
st.write("Upload a CSV with the predictor columns required for the selected prediction timing.")
template = clean_data[required].head(5)
st.download_button("Download sample input CSV", template.to_csv(index=False), "sample_prediction_input.csv", "text/csv")
uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    missing = [col for col in required if col not in new_data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model_for_upload.predict(new_data[required])
        results = new_data.copy()
        results["Prediction"] = np.where(predictions == 1, "AT RISK", "NOT AT RISK")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button("Download prediction results", results.to_csv(index=False), "student_predictions.csv", "text/csv")

st.subheader("Responsible interpretation")
st.write("Use predictions only as prompts for supportive review. Verify individual circumstances, consult teachers and learners, and avoid treating model output as a definitive judgment. The source dataset represents Portuguese secondary-school students and may not generalize to another institution.")
