import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    path = ROOT / "student_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring possible academic risk using historical UCI Mathematics records.")
model = load_model()

data_path = ROOT / "student_data.csv"
if data_path.exists():
    data = pd.read_csv(data_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical records", f"{len(data):,}")
    c2.metric("Average final grade", f"{data['G3'].mean():.2f}/20" if "G3" in data else "N/A")
    c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()) if "G3" in data else "N/A")
    with st.expander("Dataset preview"):
        st.dataframe(data.head(10), use_container_width=True)
    template = data.drop(columns=["G3"], errors="ignore").head(5)
    st.download_button("Download sample input CSV", template.to_csv(index=False).encode(),
                       "sample_prediction_input.csv", "text/csv")
else:
    st.warning("student_data.csv is not present. Run `python train_model.py` first; it downloads the UCI Mathematics data.")

if model is None:
    st.error("Trained model not found. Run `python train_model.py` before starting this app.")
    st.stop()

st.subheader("Predict uploaded student records")
st.write("Upload a CSV with the same predictor columns used during training. Do not include G3.")
uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)
        missing = [c for c in required if c not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required])
            result = new_data.copy()
            result["Prediction"] = ["AT RISK" if int(v) == 1 else "NOT AT RISK" for v in predictions]
            if hasattr(model, "predict_proba"):
                result["At-risk probability"] = model.predict_proba(new_data[required])[:, 1].round(3)
            st.dataframe(result, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))
            st.download_button("Download prediction results", result.to_csv(index=False).encode(),
                               "student_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not process the uploaded file: {e}")

st.divider()
st.caption("Responsible use: predictions are estimates, not diagnoses. A faculty member should verify context and offer supportive follow-up; never use this model as the sole basis for student decisions.")
