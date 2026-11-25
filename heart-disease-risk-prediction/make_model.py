# make_model.py - rebuild heart_model.pkl with the Python environment on THIS machine.
# Run ONCE in this folder (heart.csv must be in the same folder):   python make_model.py
#
# Why this exists: the heart_model.pkl that came with this folder was saved with
# scikit-learn 1.5.2 on the machine that trained it. Loading a pickle saved by a
# DIFFERENT scikit-learn version can fail or warn (InconsistentVersionWarning),
# because your machine has a newer sklearn. This script re-creates the exact
# notebook pipeline (same data, same seed, same steps) so the saved model matches
# the scikit-learn installed HERE - then app.py loads it cleanly.
import os

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if not os.path.exists('heart.csv'):
    raise SystemExit(
        'heart.csv is missing - put it next to make_model.py first.\n'
        '(Kaggle: fedesoriano/heart-failure-prediction, or download it from your Colab session)')

RANDOM_STATE = 42          # identical to the notebook, so the model is the same one
TARGET = 'HeartDisease'

# --- identical preprocessing to notebook Step 3 ---
df = pd.read_csv('heart.csv')
X = pd.get_dummies(df.drop(columns=[TARGET]))
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- identical winning model to notebook Step 5 (best F1-score: K-Nearest Neighbors) ---
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
metrics = {
    'accuracy':  round(float(accuracy_score(y_test, y_pred)), 3),
    'precision': round(float(precision_score(y_test, y_pred)), 3),
    'recall':    round(float(recall_score(y_test, y_pred)), 3),
    'f1':        round(float(f1_score(y_test, y_pred)), 3),
}

# --- identical bundle structure to notebook Step 8 ---
bundle = {
    'model':         model,
    'scaler':        scaler,
    'columns':       list(X.columns),
    'model_name':    'K-Nearest Neighbors',
    'metrics':       metrics,
    'healthy_means': df[df[TARGET] == 0][
        ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].mean().round(1).to_dict(),
    'dataset': {'rows': int(len(df)),
                'disease_share': float(df[TARGET].mean()),
                'source': 'Kaggle: fedesoriano/heart-failure-prediction'},
    'owner': {'name': 'Ar Kar Phyo'},
}
joblib.dump(bundle, 'heart_model.pkl')

print('Rebuilt heart_model.pkl with scikit-learn', sklearn.__version__)
print('Test metrics (should match the notebook):', metrics)
print('Now start the app:  python -m streamlit run app.py')
