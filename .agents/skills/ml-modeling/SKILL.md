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

## Leakage Rules (critical for LoL data)
- **Group by player**: one player has many matches → use `GroupKFold` / `GroupShuffleSplit` with `groups=puuid`, or the model memorizes players.
- **Same match on both sides**: a match appears for multiple top-100 players — split by `matchId` or dedupe so a game isn't in both train and test.
- **Time**: if predicting future performance, split by date (`TimeSeriesSplit`), never random.
- **No post-outcome features** when predicting a win (e.g. `gameEndedInSurrender`, final gold, `nexusKills`). Use only info available at prediction time (e.g. @15 min stats from the timeline).
- Fit scalers/encoders/imputers on train folds only → handled by `Pipeline`.

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

def make_models() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([("prep", build_preprocessor(True)),
                            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "xgb": Pipeline([("prep", build_preprocessor(False)),
                         ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                               subsample=0.8, colsample_bytree=0.8,
                                               tree_method="hist", eval_metric="logloss",
                                               random_state=RANDOM_STATE, n_jobs=-1))]),
    }

# Group-aware train/test split
X, y, groups = df[NUM_COLS + CAT_COLS], df[TARGET], df["puuid"]
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=RANDOM_STATE).split(X, y, groups))
X_tr, X_te, y_tr, y_te, g_tr = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx]

# Compare models with CV
cv = GroupKFold(n_splits=5)
for name, model in make_models().items():
    s = cross_validate(model, X_tr, y_tr, groups=g_tr, cv=cv, scoring=["roc_auc", "neg_log_loss"])
    print(f"{name}: AUC {s['test_roc_auc'].mean():.3f} ± {s['test_roc_auc'].std():.3f}")
```

## Hyperparameter Tuning (XGBoost)
```python
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "clf__n_estimators": randint(200, 1000),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__max_depth": randint(3, 8),
    "clf__min_child_weight": randint(1, 10),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__reg_lambda": loguniform(1e-2, 10),
}
search = RandomizedSearchCV(make_models()["xgb"], param_dist, n_iter=40, cv=cv,
                            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
search.fit(X_tr, y_tr, groups=g_tr)
best = search.best_estimator_
```
Tips: lower `learning_rate` + more trees generalizes better; control overfitting with `max_depth`, `min_child_weight`, `subsample`, `reg_lambda`. For imbalanced targets set `scale_pos_weight = neg/pos`.

## Early Stopping (XGBoost ≥ 2.0)
`early_stopping_rounds` is a **constructor** arg now (not a `fit` arg). Use a validation split carved from *train*, never the test set:
```python
prep = build_preprocessor(False).fit(X_fit)
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.03, early_stopping_rounds=50,
                    eval_metric="logloss", random_state=RANDOM_STATE)
xgb.fit(prep.transform(X_fit), y_fit, eval_set=[(prep.transform(X_val), y_val)], verbose=False)
print("best iteration:", xgb.best_iteration)
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

## Persist
```python
import joblib
joblib.dump(best, "models/xgb_win_v1.joblib")   # saves the whole pipeline incl. preprocessing
```

## Checklist
- [ ] Fixed `random_state` everywhere (reproducibility)
- [ ] Group/time-aware splits; no match or player in both train and test
- [ ] Beat a dummy/logistic baseline
- [ ] Report mean ± std across folds, not one score
- [ ] Test set used exactly once
- [ ] Plots follow the `data-visualization` skill
