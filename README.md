# Titanic Survival Prediction

Predicts whether a Titanic passenger survived using Logistic Regression on the Kaggle Titanic
dataset. Single-notebook learning project.

## Dataset

Kaggle ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic) files in
`data/`: `train.csv` (891 labelled passengers), `test.csv` and `gender_submission.csv`. Only
`train.csv` is used by the notebook.

## Approach (`titanic_pred.ipynb`)

- Drop `Cabin`; fill missing `Age` with the mean and `Embarked` with the mode.
- Encode `Sex` (male=0, female=1) and `Embarked` (S=0, C=1, Q=2).
- Features: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`
  (`Name`, `Ticket`, `PassengerId` dropped).
- 80/20 train/test split (`random_state=42`), default `LogisticRegression`.
- Seaborn count plots of survival overall and by sex.

Results printed in the notebook:

| Split | Accuracy |
|-------|----------|
| Train | 0.8020 |
| Test  | 0.7989 |

The notebook output also shows a `ConvergenceWarning` from the default `lbfgs` solver; scaling the
features or raising `max_iter` would address it.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook titanic_pred.ipynb
```

## Project structure

```
titanic_pred.ipynb
data/train.csv
data/test.csv
data/gender_submission.csv
```
