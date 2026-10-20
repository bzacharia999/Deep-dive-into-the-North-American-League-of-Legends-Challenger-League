---
name: ml-modeling
description: >-
  Use this skill when building, tuning, evaluating, or interpreting machine
  learning models with scikit-learn or XGBoost in this project (e.g. predicting
  wins, LP, or role from match/player features).
---

# ML Modeling Best Practices (sklearn + XGBoost)

## Workflow
1. **Define the target & metric first** (e.g. win → ROC-AUC / log loss; LP → MAE).
2. **Split before anything else.** Hold out a test set and don't touch it until the end.
3. **Baseline** (`DummyClassifier`, `LogisticRegression`) before XGBoost.
4. **All preprocessing inside a `Pipeline`** to prevent leakage.
5. **Cross-validate** with the right splitter; tune with `RandomizedSearchCV`.
6. **Evaluate once on test**, interpret (feature importance / SHAP), save model.


## Pipeline Template
```python
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "win"
NUM_COLS = ["kills", "deaths", "assists", "cs_per_min", "gold_per_min", "vision_score"]
CAT_COLS = ["role", "champion"]

def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))   # needed for linear models, not trees
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT_COLS),
    ])

# Group-aware train/test split
X, y, groups = df[NUM_COLS + CAT_COLS], df[TARGET], df["puuid"]
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=RANDOM_STATE).split(X, y, groups))
X_tr, X_te, y_tr, y_te, g_tr = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx]
```

## Final Evaluation & Interpretation
```python
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

proba = best.predict_proba(X_te)[:, 1]
print("Test AUC:", round(roc_auc_score(y_te, proba), 3))
print(classification_report(y_te, best.predict(X_te)))
RocCurveDisplay.from_estimator(best, X_te, y_te)
ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, normalize="true")

# Model-agnostic importance on raw columns (preferred over gain-based importance)
pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()

# SHAP for XGBoost (optional): shap.TreeExplainer(best["clf"]).shap_values(best["prep"].transform(X_te))
```

## Checklist
- [ ] Fixed `random_state` everywhere (reproducibility)
- [ ] Beat a dummy/logistic baseline
- [ ] Report mean ± std across folds, not one score
- [ ] Test set used exactly once
