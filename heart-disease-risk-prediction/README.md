# Heart Disease Risk Prediction System

An end-to-end, educational machine-learning project: 918 public patient records ->
one clean pipeline (one-hot encoding, train-only scaling, seed-42 stratified split) ->
three classifiers compared on F1 -> a K-Nearest Neighbours screening assistant ->
a small Streamlit web app that explains each prediction in plain language.

> **Educational prototype - not a medical device.** It is a screening *aid* that ranks
> who may need further testing first. It never diagnoses anything and must not be used
> for real clinical decisions.

## Results (184 unseen test patients, seed 42)

| Model | Accuracy | Precision | Recall | F1 | Train accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.886 | 0.872 | 0.931 | 0.900 | 0.862 |
| Decision Tree | 0.793 | 0.814 | 0.814 | 0.814 | 1.000 |
| **K-Nearest Neighbors (selected)** | **0.918** | **0.899** | **0.961** | **0.929** | 0.872 |

Confusion matrix of the selected model: **71** true negatives, **11** false positives,
**4** false negatives, **98** true positives. Recall is the priority metric here: in
screening, a missed sick patient costs more than one extra follow-up. Every headline
number is simple arithmetic on that matrix - e.g. recall = 98 / (98 + 4) = 0.961.

## Repository layout

```
heart-disease-risk-prediction/
├── Heart_Disease_Risk_Prediction_System.ipynb   # full pipeline, Steps 1-9
├── app.py                                       # Streamlit web app (3 tabs)
├── make_model.py                                # rebuilds heart_model.pkl deterministically
├── start_app.bat                                # one double-click: install -> rebuild -> run
├── requirements.txt                             # pinned, tested versions
├── heart.csv                                    # the dataset (918 rows, 11 features + label)
├── .streamlit/config.toml                       # light clinical theme
├── LICENSE                                      # MIT
└── README.md
```

No trained model file is committed on purpose: `heart_model.pkl` is rebuilt locally so
it always matches *your* scikit-learn version (see `make_model.py`).

## Quickstart

**Windows - the easy way:** double-click `start_app.bat`. It installs the packages
(first run only), rebuilds the model file for this machine's scikit-learn, and starts
the app. Your browser opens at `localhost:8501`.

**Any OS - from a terminal:**

```bash
python -m pip install -r requirements.txt
python make_model.py          # rebuilds heart_model.pkl (prints the test metrics)
python -m streamlit run app.py
```

Python 3.12+ recommended (tested with the pinned versions in `requirements.txt`).

In the app: the **Risk assessment** tab holds the intake form with two ready-made
example patients (a high-risk preset predicts 100 percent, a healthy preset 0 percent)
and lists the factors that pushed the decision. The **Model card** tab documents
metrics and pipeline; **About the data** shows dataset facts and a plain-language
dictionary.

## The notebook

`Heart_Disease_Risk_Prediction_System.ipynb` is the single source of truth - every
number in the app comes from it:

1. **Steps 1-2** - data acquisition through the official Kaggle API (`kagglehub`),
   cleaning, exploratory analysis.
2. **Step 3** - preprocessing: one-hot encoding (11 -> 20 columns), re-indexing to the
   training column contract, StandardScaler fitted on training rows only, stratified
   80/20 split with `random_state=42`.
3. **Steps 4-5** - training three classifiers (Logistic Regression, Decision Tree,
   KNN) and evaluation: confusion matrix, accuracy, precision, recall, F1.
4. **Steps 6-6b** - single-patient prediction plus an optional API-written
   explanation (Google Gemini, free tier - set the `GEMINI_API_KEY` environment
   variable to enable; the notebook runs fine without it).
5. **Step 7** - responsible-AI reflection.
6. **Steps 8-9** - packaging the bundle (`heart_model.pkl`) and writing the web-app
   files.

Run it top-to-bottom in Jupyter, VS Code or Google Colab. Note that the notebook
re-generates `app.py` and `requirements.txt` next to itself - the copies in this
repository are identical to what it writes.

## The dataset

- Source: [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
  by fedesoriano on Kaggle - a compilation of five public cardiac datasets
  (Cleveland, Hungarian, Switzerland, Long Beach VA, and a synthetic portion).
- 918 patient records, no missing values, 11 routine clinical features
  (age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood
  sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, ST slope)
  plus the `HeartDisease` label (410 healthy / 508 disease).
- `heart.csv` is included here so the app works out of the box; for redistribution
  beyond this educational context, please refer to the terms on the dataset page.

## Reproducibility

- Fixed `random_state=42` everywhere (split, models).
- The scaler is fitted on training data only - no leakage.
- Model selection is F1-driven, favoring the recall/precision balance a screening
  tool needs.
- `make_model.py` rebuilds the bundle deterministically and prints the test metrics;
  they match the notebook exactly.

## Limitations & responsible use

- The KNN "probability" is a neighbour vote share - read it as a ranking of who
  needs testing first, not as a medical percentage.
- Trained on a mostly-adult, mixed-origin public dataset; it will not generalize to
  every population.
- False negatives are possible (4 of 102 sick patients in the test set) - in a real
  screening program the operating point and human review would be set by clinicians.

## License

Released under the [MIT License](LICENSE). The dataset has its own terms on Kaggle
(see above).
