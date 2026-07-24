# HeartAid — Heart Disease Risk Prediction

A machine learning project that predicts a patient's risk of heart disease from routine clinical measurements (age, blood pressure, cholesterol, chest pain type, and more), built as an end-to-end classification pipeline: problem framing → data cleaning → EDA → feature engineering → model comparison → evaluation.

## The Business Question

Given a patient's basic clinical measurements, can we flag who is at risk of heart disease *before* more invasive testing is needed? A model like this isn't meant to replace a doctor's diagnosis — it's meant to help prioritize who should get a closer look, which matters because missing an at-risk patient is a far more costly mistake than sending a healthy patient for an unnecessary follow-up.

That framing — false negatives are worse than false positives — shaped every modeling decision in this project.

## Dataset

**Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle, fedesoriano)
**Size:** 918 patient records, 11 clinical features, 1 binary target (`HeartDisease`)

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | M / F |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of the peak exercise ST segment (Up/Flat/Down) |
| **HeartDisease** | **Target** — 1 = heart disease present, 0 = no heart disease |

Class balance: 508 positive cases vs. 410 negative cases (~55/45) — mild imbalance, not severe enough to require resampling techniques, but stratified splitting was used to keep it consistent across train/test.

## Data Cleaning

`.isnull().sum()` reported zero missing values across the entire dataset — but that was misleading. Closer inspection of `.describe()` revealed physically impossible values disguised as real numbers:

- **`RestingBP`**: 1 row had a value of `0` (a living patient cannot have zero blood pressure). Replaced with the column median.
- **`Cholesterol`**: 172 rows (~19% of the dataset) had a value of `0`. Given the scale of the problem, dropping these rows wasn't viable — it would have discarded a fifth of the data. Instead, missing cholesterol values were imputed using the **median, computed separately for each `HeartDisease` class**, since patients with heart disease had a noticeably higher median cholesterol (246) than healthy patients (231.5). A single global median would have blurred that real signal. Median (not mean) was used because the cholesterol distribution is right-skewed (max of 603 vs. a 75th percentile of only 275).

## Exploratory Data Analysis

Key patterns worth noting:
- `Cholesterol` correlates with heart disease risk more strongly once the disguised zero-values are properly imputed.
- Class distribution is close to balanced, simplifying the modeling approach (no SMOTE or class-weighting needed).
- Categorical features vary in nature: `Sex` and `ExerciseAngina` are binary; `ChestPainType` and `RestingECG` are unordered categories; `ST_Slope` (Up/Flat/Down) has a natural clinical ordering, though it was one-hot encoded like the rest for simplicity in this pass.

## Feature Engineering

- One-hot encoded all 5 categorical columns (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) using `drop_first=True` to avoid redundant columns.
- Final feature set: 15 columns (from the original 11).
- 80/20 train/test split, **stratified on the target** to preserve class balance in both sets.
- Features were standardized (`StandardScaler`) for the distance-based/linear models (Logistic Regression, SVM); tree-based models (Random Forest, Gradient Boosting, XGBoost) were trained on unscaled data, since they split on per-feature thresholds and are insensitive to feature scale.

## Model Comparison

Five models were trained and evaluated honestly against each other, rather than assuming a more complex model would automatically perform better:

| Model | Accuracy | False Negatives | False Positives |
|---|---|---|---|
| **Gradient Boosting** | **89.7%** | **9** | 10 |
| Logistic Regression | 88.0% | 10 | 12 |
| Random Forest | 85.9% | 13 | 13 |
| SVM | 85.3% | 11 | 16 |

*(XGBoost also evaluated as an additional comparison — see notes below.)*

### Why Gradient Boosting was chosen

Gradient Boosting achieved the best result on every metric that matters for this problem: highest accuracy, and — most importantly — the fewest false negatives (missed diagnoses). In this domain, a false negative means a patient with real heart disease is told they're healthy, which is a far more serious error than a false positive (an unnecessary follow-up test). Gradient Boosting builds trees sequentially, with each new tree specifically correcting the errors of the previous ones, which let it pick up on patterns the other models missed without overfitting as heavily as Random Forest did on this relatively small dataset (734 training rows).

**A note on rigor:** these results come from a single train/test split, and the margin between the top models (Gradient Boosting vs. Logistic Regression) is a small number of patients — a different random seed could plausibly shift the ranking slightly. Logistic Regression remains a strong, more interpretable runner-up, and would be the preferred choice in a setting where model explainability (e.g. showing a doctor exactly which factors drove a prediction) outweighs a small accuracy gain.

## Confusion Matrix — Reading the Result

Gradient Boosting's confusion matrix:
```
[[72 10]
 [ 9 93]]
```
- **72** patients correctly identified as healthy
- **93** patients correctly identified as at-risk
- **10** healthy patients incorrectly flagged as at-risk (false alarms — costs extra testing, not a missed diagnosis)
- **9** at-risk patients incorrectly predicted as healthy (the costliest error type in this context)

## How to Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install pandas scikit-learn matplotlib seaborn xgboost joblib

python model.py
```

## Project Structure

```
HeartAid/
├── README.md
├── model.py              # full ML pipeline: load → clean → encode → train → evaluate
├── dataset/
│   └── heart.csv
└── models/
    └── heart_disease_model.joblib
```

## Key Takeaways

- Missing data isn't always `NaN` — disguised placeholders (like `0` standing in for "unknown") require checking `.describe()`, not just `.isnull()`.
- The "best" imputation strategy should be justified by what the data actually shows (skew → median over mean; group differences → per-group imputation over a global average).
- More complex models aren't automatically better — model choice should be validated by comparing real metrics, not assumed from reputation.
- In a medical risk-prediction context, the choice of evaluation metric matters as much as the model itself: optimizing for accuracy alone can hide a model that's quietly missing more real cases than a "less accurate" alternative.