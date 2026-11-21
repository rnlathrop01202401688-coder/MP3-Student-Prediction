# LMS Student Rating Predictor — Machine Problem 3

This Streamlit application uses the eight LMS activity fields from Machine Problem 2 to predict **High/Low rating**. The dataset does not provide a verified Pass/Fail label.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `python train_model.py` if you want to regenerate `model.joblib` and `metrics.json`.
5. Run `streamlit run app.py`.

## Deploy with Streamlit Community Cloud

1. Upload the project files to a GitHub repository (do not upload identifiable student data).
2. Visit https://share.streamlit.io and connect the repository.
3. Set the entrypoint to `app.py`, then deploy.
4. Use an approved private hosting solution for real student information; public Community Cloud apps are not appropriate for confidential student records.

## Notes

- The model is trained with `RandomForestClassifier(n_estimators=300, random_state=42)` on a stratified 80/20 split.
- The saved model is loaded at runtime, so the hosted app does not need to retrain on every page load.
- Class probabilities are uncalibrated; do not present them as validated confidence.
- The dataset filename indicates possible augmentation. Evaluate with truly independent student/course data before real academic use.
