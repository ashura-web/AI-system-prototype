# %%writefile app.py
# Heart Disease Risk Prediction System - Streamlit web app
# Educational research prototype by Ar Kar Phyo.
#
# UI (v2): light clinical theme, wide layout, no sidebar. Three tabs:
#   1. Risk assessment - intake form (left) + risk report panel (right)
#   2. Model card      - metrics, pipeline description, author
#   3. About the data  - dataset facts, healthy-group averages, data dictionary
# Colors live in .streamlit/config.toml. Two preset buttons load demo patients
# (high-risk: 100%, healthy: 0% - verified against the packaged model).
# The prediction pipeline is EXACTLY the notebook's (Steps 3-6): one-hot encode,
# reindex to the training columns, StandardScaler, K-Nearest Neighbors.
# Run: streamlit run app.py  (or double-click start_app.bat) -> http://localhost:8501
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Heart Disease Risk Prediction System', layout='wide')

# ---------- styling: own CSS classes only, so nothing breaks between versions ----------
CSS = '''
<style>
.block-container {max-width: 1080px; padding-top: 1.4rem;}
.hero {background: linear-gradient(130deg, #0F766E 0%, #115E59 55%, #134E4A 100%);
       border-radius: 18px; padding: 1.5rem 2rem 1.3rem 2rem; color: #F0FDFA;}
.hero h1 {margin: 0; font-weight: 800; font-size: 1.85rem; color: #F0FDFA;
          letter-spacing: .01em;}
.hero p {margin: .4rem 0 0 0; color: #99F6E4; font-size: .98rem;}
.pill {display: inline-block; background: rgba(255,255,255,.13);
       border: 1px solid rgba(255,255,255,.28); color: #CCFBF1;
       padding: .14rem .75rem; border-radius: 999px; margin: .7rem .45rem 0 0;
       font-size: .8rem;}
.badge {border-radius: 18px; padding: 1.15rem 1.3rem; text-align: center; color: #FFFFFF;}
.badge.high {background: linear-gradient(135deg, #DC2626, #B91C1C);}
.badge.low  {background: linear-gradient(135deg, #059669, #047857);}
.badge .pct {font-size: 2.5rem; font-weight: 800; line-height: 1.05;}
.badge .lbl {font-size: .95rem; opacity: .93; margin-top: .25rem;}
.chip {display: inline-block; background: #FFFBEB; border: 1px solid #FCD34D;
       color: #92400E; padding: .18rem .75rem; border-radius: 999px;
       margin: .18rem .3rem .18rem 0; font-size: .84rem;}
.note {background: #F1F5F9; border-left: 4px solid #0F766E; border-radius: 8px;
       padding: .75rem 1rem; color: #334155; font-size: .92rem;}
</style>'''
st.markdown(CSS, unsafe_allow_html=True)

# ---------- load the packaged pipeline (notebook Steps 3-6), cached for the session ----------
@st.cache_resource
def load_bundle():
    if not os.path.exists('heart_model.pkl'):
        st.error('heart_model.pkl not found - run the notebook up to Step 8 first, '
                 'then put the file next to app.py.')
        st.stop()
    return joblib.load('heart_model.pkl')

bundle = load_bundle()
model, scaler = bundle['model'], bundle['scaler']
EXPECTED_COLUMNS = bundle['columns']
m = bundle['metrics']
hm = bundle['healthy_means']

# ---------- preset patients for the demo (probabilities verified against the model) ----------
PRESET_HIGH = {'Age': 54, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 140,
               'Cholesterol': 239, 'FastingBS': 0, 'RestingECG': 'ST', 'MaxHR': 160,
               'ExerciseAngina': 'Y', 'Oldpeak': 1.2, 'ST_Slope': 'Flat'}          # -> 100%
PRESET_HEALTHY = {'Age': 38, 'Sex': 'F', 'ChestPainType': 'ATA', 'RestingBP': 120,
                  'Cholesterol': 180, 'FastingBS': 0, 'RestingECG': 'Normal',
                  'MaxHR': 175, 'ExerciseAngina': 'N', 'Oldpeak': 0.0,
                  'ST_Slope': 'Up'}                                                # -> 0%

# ---------- prediction: identical pipeline to the notebook ----------
def predict(age, sex, cpt, rbp, chol, fbs, ecg, maxhr, ang, oldpeak, slope):
    patient = pd.DataFrame([{'Age': age, 'Sex': sex, 'ChestPainType': cpt,
                             'RestingBP': rbp, 'Cholesterol': chol, 'FastingBS': fbs,
                             'RestingECG': ecg, 'MaxHR': maxhr,
                             'ExerciseAngina': ang, 'Oldpeak': oldpeak,
                             'ST_Slope': slope}])
    enc    = pd.get_dummies(patient).reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    scaled = scaler.transform(enc)
    prob   = float(model.predict_proba(scaled)[0][1])
    return prob

def standout_factors(age, rbp, chol, maxhr, oldpeak, fbs, ang, slope, cpt):
    # plain-language flags, computed from the healthy-group averages in the bundle
    notes = []
    checks = [('Age', age, 1.10, 'above'), ('RestingBP', rbp, 1.10, 'above'),
              ('Cholesterol', chol, 1.10, 'above'),
              ('Oldpeak', oldpeak, 1.10, 'above'),
              ('MaxHR', maxhr, 0.90, 'below')]
    for key, value, factor, word in checks:
        if value > hm[key] * factor and word == 'above':
            notes.append(f'{key} ({value:g}) is well above the healthy-group average ({hm[key]:g})')
        if value < hm[key] * factor and word == 'below':
            notes.append(f'{key} ({value:g}) is well below the healthy-group average ({hm[key]:g})')
    if fbs == 1:
        notes.append('Fasting blood sugar above 120 mg/dl')
    if ang == 'Y':
        notes.append('Exercise-induced angina')
    if slope in ('Flat', 'Down'):
        notes.append(f'Abnormal exercise ST slope ({slope})')
    if cpt == 'ASY':
        notes.append('Asymptomatic chest pain presentation (ASY)')
    return notes

# ---------- hero banner ----------
owner, d = bundle['owner'], bundle['dataset']
st.markdown(
    '<div class="hero">'
    '<h1>Heart Disease Risk Prediction System</h1>'
    f'<p>A K-Nearest Neighbours screening assistant, trained on {d["rows"]} public patient '
    'records - enter a patient profile and see what the model would flag.</p>'
    f'<span class="pill">{owner["name"]}</span>'
    
    '<span class="pill">Educational prototype - not a medical device</span>'
    '</div>', unsafe_allow_html=True)
st.write('')

tab_predict, tab_model, tab_data = st.tabs(['Risk assessment', 'Model card', 'About the data'])

# ============================== tab 1: risk assessment ==============================
with tab_predict:
    left, right = st.columns([3, 2], gap='large')

    # ----- left: intake form -----
    with left:
        st.markdown('##### Patient intake form')
        p1, p2 = st.columns(2)
        if p1.button('Load high-risk example', use_container_width=True):
            st.session_state.update(PRESET_HIGH)
            st.rerun()
        if p2.button('Load healthy example', use_container_width=True):
            st.session_state.update(PRESET_HEALTHY)
            st.rerun()

        st.markdown('**Demographics**')
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider('Age (years)', 20, 100, 54, key='Age')
        with c2:
            sex = st.radio('Sex', ['M', 'F'], horizontal=True, key='Sex')

        st.markdown('**Vitals and blood work**')
        c1, c2, c3 = st.columns(3)
        with c1:
            rbp = st.slider('Resting blood pressure (mm Hg)', 80, 200, 140, key='RestingBP')
        with c2:
            chol = st.slider('Cholesterol (mm/dl)', 0, 600, 239, key='Cholesterol')
        with c3:
            fbs = st.radio('Fasting blood sugar > 120 mg/dl', [0, 1], horizontal=True,
                           format_func=lambda v: 'No' if v == 0 else 'Yes', key='FastingBS')

        st.markdown('**Exercise and ECG findings**')
        c1, c2 = st.columns(2)
        with c1:
            maxhr = st.slider('Max heart rate (bpm)', 60, 210, 160, key='MaxHR')
            ang = st.radio('Exercise-induced angina', ['N', 'Y'], horizontal=True,
                           key='ExerciseAngina')
            oldpeak = st.slider('Oldpeak - ST depression during exercise', -3.0, 7.0, 1.2,
                                0.1, key='Oldpeak')
        with c2:
            ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'], key='RestingECG')
            cpt = st.selectbox('Chest pain type', ['ATA', 'NAP', 'ASY', 'TA'],
                               help='ATA: atypical angina, NAP: non-anginal pain, '
                                    'ASY: asymptomatic, TA: typical angina',
                               key='ChestPainType')
            slope = st.selectbox('ST slope', ['Up', 'Flat', 'Down'], key='ST_Slope')

    # ----- right: risk report -----
    with right:
        st.markdown('##### Risk report')
        clicked = st.button('Predict risk', type='primary', use_container_width=True)
        if clicked:
            prob = predict(age, sex, cpt, rbp, chol, fbs, ecg, maxhr, ang, oldpeak, slope)
            st.session_state['result'] = {'prob': prob,
                                          'notes': standout_factors(age, rbp, chol, maxhr,
                                                                    oldpeak, fbs, ang,
                                                                    slope, cpt)}
        result = st.session_state.get('result')
        if result is None:
            st.markdown('<div class="note">Fill in the patient values on the left (or load '
                        'an example), then press <b>Predict risk</b>. The result appears '
                        'here and stays until you predict again.</div>',
                        unsafe_allow_html=True)
        else:
            prob = result['prob']
            is_high = prob >= 0.5
            cls = 'high' if is_high else 'low'
            label = 'HIGH RISK of heart disease' if is_high else 'LOW risk of heart disease'
            st.markdown(
                f'<div class="badge {cls}"><div class="pct">{prob:.0%}</div>'
                f'<div class="lbl">{label}</div></div>', unsafe_allow_html=True)
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f'The {bundle["model_name"]} decision boundary sits at 50% - this '
                       f'patient is on the {"disease" if is_high else "healthy"} side of it.')
            st.markdown('**Values that stand out against the healthy group:**')
            if result['notes']:
                chips = ''.join(f'<span class="chip">{n}</span>' for n in result['notes'])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption('No single value stands out strongly against the healthy group.')
            with st.expander('How this prediction is made'):
                st.markdown(
                    f"The {bundle['model_name']} was trained in the notebook on "
                    f"{d['rows']} patients (80/20 stratified split, StandardScaler fitted on "
                    'the training set only). This app rebuilds that exact pipeline: one-hot '
                    'encoding -> column reindexing -> scaling -> prediction, using the '
                    'objects packaged in heart_model.pkl.')
                st.caption('Screening support only - a doctor must confirm any result.')

# ============================== tab 2: model card ==============================
with tab_model:
    st.markdown('##### Model card')
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Test accuracy', f"{m['accuracy']:.1%}")
    k2.metric('Precision', f"{m['precision']:.1%}")
    k3.metric('Recall', f"{m['recall']:.1%}")
    k4.metric('F1-score', f"{m['f1']:.2f}")
    st.markdown(
        f"**Algorithm** - {bundle['model_name']}, chosen in notebook Step 5 by the best "
        'F1-score against Decision Tree and Logistic Regression baselines. '
        f"**Training data** - {d['rows']} patients, {d['disease_share']:.0%} with heart "
        f"disease ({d['source']}).")
    st.markdown('**Pipeline in this app**')
    st.markdown(
        '1. the patient dict is one-hot encoded exactly like the training data\n\n'
        '2. columns are re-indexed to the training columns (missing dummies become 0)\n\n'
        '3. the StandardScaler packaged in Step 8 transforms the row\n\n'
        '4. the packaged KNN returns the probability of the disease class')
    st.markdown(f"**Author** - {owner['name']}. "
                'This web app is the communication layer of the notebook project; '
                'it never re-trains anything.')

# ============================== tab 3: about the data ==============================
with tab_data:
    st.markdown('##### About the data')
    st.markdown(
        f"The model was trained on {d['rows']} patient records published on Kaggle "
        f"({d['source']}); {d['disease_share']:.0%} of them have heart disease. "
        'The averages below come from the healthy group of that dataset and are what '
        'the risk report compares your patient against.')
    hm_table = pd.DataFrame({'Feature': list(hm.keys()),
                             'Healthy-group average': [f'{v:g}' for v in hm.values()]})
    st.dataframe(hm_table, hide_index=True, use_container_width=True)
    st.markdown('##### The 11 input features')
    dictionary = pd.DataFrame([
        ('Age', 'years of the patient'),
        ('Sex', 'M = male, F = female'),
        ('ChestPainType', 'ATA = atypical angina, NAP = non-anginal pain, '
         'ASY = asymptomatic, TA = typical angina'),
        ('RestingBP', 'resting blood pressure in mm Hg'),
        ('Cholesterol', 'serum cholesterol in mm/dl'),
        ('FastingBS', '1 if fasting blood sugar > 120 mg/dl, else 0'),
        ('RestingECG', 'Normal, ST (ST-T abnormality), LVH (left ventricular hypertrophy)'),
        ('MaxHR', 'maximum heart rate achieved, in bpm'),
        ('ExerciseAngina', 'Y = angina induced by exercise, N = no'),
        ('Oldpeak', 'ST depression measured during exercise relative to rest'),
        ('ST_Slope', 'the slope of the peak exercise ST segment: Up, Flat, Down'),
    ], columns=['Feature', 'Meaning'])
    st.dataframe(dictionary, hide_index=True, use_container_width=True)

st.caption('Educational prototype - not a medical device. Predictions do not replace '
           'a doctor.')
