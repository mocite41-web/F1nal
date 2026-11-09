# Student Performance Prediction System

An educational Streamlit prototype using the UCI Student Performance Mathematics dataset.

## Features
- Predicts `AT RISK` when G3 < 10; otherwise `NOT AT RISK`.
- Compares Random Forest and Logistic Regression.
- Uses the same stratified 80/20 split settings (`random_state=42`) for both models.
- Displays accuracy, precision, recall, F1-score, and confusion matrices.
- Discusses Random Forest feature importance and Logistic Regression coefficients.
- Repeats experiments without G1 and G2 for earlier-cycle prediction.
- Accepts CSV uploads and provides downloadable predictions.

## 1. Setup locally
Python 3.10 or later is recommended.

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

`train_model.py` downloads the dataset from the UCI repository and creates `student_data.csv`.
Open the local URL printed by Streamlit, usually http://localhost:8501.

## 2. Publish to GitHub
Create a public repository named `student-performance-prediction`. In the project folder:

```bash
git init
git add app.py train_model.py requirements.txt README.md student_data.csv
git commit -m "Add model comparison and early prediction experiments"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not commit passwords, API tokens, or private student information.

## 3. Deploy with Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app**.
3. Choose the `student-performance-prediction` repository and `main` branch.
4. Set the main file path to `app.py`.
5. Select **Deploy** and wait for the build to finish.
6. Streamlit will provide a public app URL. Test the dashboard and CSV upload using that URL.

Keep `app.py`, `requirements.txt`, and `student_data.csv` in the repository root. If the dataset is not committed, the app cannot load until it is supplied. The app trains the four model variants when it first loads, so initial startup may take a little time.

## Experimental design
- Target: `(G3 < 10).astype(int)`.
- Class 1: At risk; Class 0: Not at risk.
- Full feature set: all predictors except G3.
- Early feature set: all predictors except G3, G1, and G2.
- Split: 80% training / 20% testing, stratified, `random_state=42`.
- Models: Random Forest (300 trees, balanced class weights) and Logistic Regression (balanced class weights).
- The split is performed separately for the two feature sets with identical parameters. The row partition is therefore consistent because the same data, target, and random state are used.

## Interpreting results
Prioritize recall for the `At risk` class when considering whether the model misses learners who may need support. Also examine precision and F1-score because a high recall can come with more false alarms. Feature importance and coefficients indicate associations learned by the model, not causes.

## Limitations and responsible use
This is a classroom demonstration using public historical data. The dataset represents students from Portuguese secondary schools and may not generalize to other schools or populations. Predictions are not diagnoses and must not be used as the sole basis for real student decisions. Human review and supportive intervention are essential.
