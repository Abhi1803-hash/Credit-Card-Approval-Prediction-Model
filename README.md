# Credit Card Approval Prediction

A logistic regression model that predicts whether a credit card application will be approved, built on the UCI Credit Approval dataset.

**Result:** 84.2% test accuracy, and 85.1% mean accuracy in 5-fold cross-validated grid search.

> This was a guided project completed as part of my [DataCamp](https://app.datacamp.com/learn) learning. The *Known issues* section below covers what I'd fix now.

---

## Problem

Banks receive far more credit card applications than analysts can review by hand. An automated first pass can approve clear cases quickly and route borderline ones to a human, cutting review time and making decisions more consistent.

## Data

[UCI Credit Approval (crx)](https://archive.ics.uci.edu/dataset/27/credit+approval): **690 applications, 15 features**, plus an approved (`+`) or rejected (`-`) label. Feature names are anonymised to protect applicants. Based on public analyses of this dataset, they likely include age, debt, years employed, credit score, income and prior default.

## Approach

1. Replaced `?` placeholders with `NaN` and imputed missing values.
2. Label-encoded the categorical features.
3. Dropped two weak features (driver's licence, zip code).
4. Split the data 67/33 into train and test sets, and scaled features to 0–1 with `MinMaxScaler`.
5. Trained a logistic regression model.
6. Tuned `tol` and `max_iter` with `GridSearchCV` (5-fold).

## Results

| Metric | Value |
|---|---|
| Test accuracy | **84.2%** |
| Best CV accuracy (grid search) | **85.1%** (`tol=0.01`, `max_iter=100`) |

Confusion matrix on the test set (228 applications):

| | Predicted reject | Predicted approve |
|---|---|---|
| **Actually rejected** | 94 | 9 |
| **Actually approved** | 27 | 98 |

The model rarely approves someone it should reject: only 9 false approvals, which is the costly mistake for a bank. It is more cautious the other way, wrongly rejecting 27 applicants who should have been approved.

## Known issues and what I'd improve

Reviewing this project later, I found three issues:

1. **Missing ages are filled with the letter "b".** The categorical imputation loop calls `fillna` on the *whole* DataFrame with the most common value of the first column. So the 12 missing ages become `"b"`, and age, which is stored as text, gets label-encoded as a category instead of being treated as a number. **Fix:** convert age to numeric, then impute each column with its own median or mode.
2. **Data leakage.** Imputation happens before the train/test split, and the grid search scales the full dataset before cross-validation. Both let test-set information leak into training. **Fix:** put the imputer, encoder and scaler inside a scikit-learn `Pipeline`, so each step is fit on training folds only.
3. **Accuracy alone is a weak metric for credit decisions.** **Fix:** report precision, recall and ROC-AUC, and pick a decision threshold based on the cost of a bad approval versus a lost customer.

My [Churn Prediction Pipeline](https://github.com/Abhi1803-hash/Churn-Prediction-Pipeline) project applies these fixes: a leak-free `Pipeline`, proper encoding, and ROC-AUC and F1 reporting.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook credit-card-approval-predictions.ipynb
```

The notebook was written on Kaggle. To run it locally, change the `read_csv` path in the first cell to `crx.csv`.

## Tech stack

Python · pandas · NumPy · scikit-learn (`LogisticRegression`, `MinMaxScaler`, `GridSearchCV`)
