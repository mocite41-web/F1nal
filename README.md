# Student Performance Prediction System

## Project purpose
A classroom prototype that estimates whether a Mathematics student may be at risk of failing the final grade using the UCI Student Performance dataset.

**Target rule:** `G3 < 10` = `AT RISK`; `G3 >= 10` = `NOT AT RISK`. `G3` is the outcome only and is never a predictor.

## Files
- `train_model.py` — downloads Mathematics data, prepares features, compares Random Forest and Logistic Regression on the same stratified split, evaluates both, creates confusion matrices, calculates permutation feature importance, and repeats without G1/G2.
- `app.py` — Streamlit interface for uploading student records and downloading predictions.
- `requirements.txt` — Python dependencies.
- `student_data.csv` — created automatically by `train_model.py` after dataset retrieval.
- `student_data_template.csv` — header-only input template; use the app's sample CSV download for real example rows after training.
- Evaluation CSVs and PNG visualizations — created when training completes.

## Setup (Python 3.10+)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
The first training run requires internet access to retrieve the UCI dataset. It writes `student_data.csv` and the model/evaluation outputs into this folder.

## GitHub
Create a public repository named `student-performance-prediction`, then:
```bash
git init
git add .
git commit -m "Build student performance prediction system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME`. Do not commit passwords or access tokens.

## Streamlit Community Cloud deployment
1. Push this project to GitHub.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch `main`, and set the main file path to `app.py`.
4. Deploy. The app expects `student_model.pkl` and `student_data.csv` to exist in the repository. Run `python train_model.py` locally first, then commit the generated files if your instructor permits model/data files in GitHub.
5. If repository size or course policy prohibits committing generated artifacts, adapt the app to train/download artifacts during deployment or store them in an approved external location.

## Experiments and evaluation
- Main experiment: Random Forest and Logistic Regression use identical `train_test_split` settings (`test_size=0.20`, `random_state=42`, `stratify=y`).
- Early prediction experiment: same split and models, but both `G1` and `G2` are removed.
- Metrics: accuracy, precision, recall, F1-score. Confusion matrices are saved as PNG files.
- Feature importance: held-out permutation importance using F1 scoring. It measures the impact of shuffling each original input column; it is not causal evidence.

## Responsible use and limitations
This is a classroom demonstration on historical records from two Portuguese secondary schools. It may not generalize to another school, country, curriculum, or student population. Predictions are not diagnoses and must not be used as the sole basis for decisions. A faculty member should verify the student's situation and offer supportive intervention.

## Dataset citation
Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
