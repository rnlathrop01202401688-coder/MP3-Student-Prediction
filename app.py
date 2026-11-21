from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='LMS Student Rating Predictor', page_icon='🎓', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model_compressed.joblib')

bundle = load_model()
model, features = bundle['model'], bundle['features']
metrics = json.loads((ROOT / 'metrics.json').read_text())

st.title('🎓 LMS Student Rating Predictor')
st.write('Enter LMS activity counts for one student/course record to estimate its **High** or **Low** rating.')
st.info('This model predicts LMS rating, not confirmed pass/fail status. Use it as a screening signal for supportive follow-up, not for automatic grading.')

labels = {
    'Submission.Actions':'Submission actions',
    'Quiz.Actions':'Quiz actions',
    'Forum.Actions':'Forum actions',
    'Create.Actions':'Create actions',
    'Read.Actions':'Read actions',
    'Update.Actions':'Update actions',
    'Delete.Actions':'Delete actions',
    'Days.Logged.In':'Days logged in',
}
with st.form('prediction_form'):
    st.subheader('Student LMS activity')
    values = {}
    for feature in features:
        values[feature] = st.number_input(labels.get(feature,feature), min_value=0, max_value=1000000, value=0, step=1)
    submitted = st.form_submit_button('Predict student rating', type='primary')

if submitted:
    row = pd.DataFrame([{name:values[name] for name in features}],columns=features)
    predicted = model.predict(row)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(row)[0]))
    if predicted == 'Low':
        st.warning('Predicted LMS rating: Low — consider a supportive check-in.')
    else:
        st.success('Predicted LMS rating: High')
    st.caption('Model-estimated class probabilities are not calibrated confidence scores.')
    st.bar_chart(pd.DataFrame({'Probability':probabilities}))

with st.expander('Model evaluation and limitations'):
    st.metric('Held-out test accuracy', f"{metrics['accuracy']:.2%}")
    st.write(f"Training records: {metrics['training_rows']:,} | Test records: {metrics['test_rows']:,}")
    st.write('Random Forest, 300 trees, stratified 80/20 split, random state 42.')
    st.write('Feature importance (relative contribution within this model):')
    st.bar_chart(pd.Series(metrics['feature_importances']).sort_values(ascending=False))
    st.caption('The provided dataset may include augmented records. Test performance on future independent students is not established; duplicate/student-level leakage has not been ruled out. LMS activity can reflect access and connectivity barriers.')
