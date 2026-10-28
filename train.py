"""DiaPredict - training script.

Run:  python train.py
Needs: data/diabetes.csv (Pima Indians Diabetes dataset, Outcome column = target)
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # falls back to sklearn's Gradient Boosting
    HAS_XGB = False

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = Path("data/diabetes.csv")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# In these columns a 0 is physiologically impossible, so it really means "missing".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    if not set(FEATURES + ["Outcome"]).issubset(df.columns):
        if df.shape[1] != 9:
            raise ValueError(f"Expected 9 columns, found {df.shape[1]}: {list(df.columns)}")
        try:
            float(df.columns[0])  # first data row was read as the header
            df = pd.read_csv(DATA_PATH, header=None)
        except ValueError:
            pass  # has a header, just with different names
        df.columns = FEATURES + ["Outcome"]
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df[FEATURES], df["Outcome"]


def make_pipe(model):
    # Imputer and scaler live INSIDE the pipeline, so they are fit only on
    # training data (and only on the training part of each CV fold).
    # This is what prevents preprocessing leakage (Syllabus 1.3).
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def build_search_space():
    spaces = {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
            {"model__penalty": ["l1", "l2"], "model__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=True, random_state=SEED),
            {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }
    if HAS_XGB:
        spaces["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=SEED),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
            },
        )
    else:
        spaces["Gradient Boosting"] = (
            GradientBoostingClassifier(random_state=SEED),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
            },
        )
    return spaces


def evaluate(name, model, X_test, y_test, cv_auc):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (5-fold)": cv_auc,
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    cm = confusion_matrix(y_test, pred).tolist()
    return metrics, roc, cm


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models, best_params = {}, {}
    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")

    # 1) Tune each base model with cross-validated grid search
    for name, (estimator, grid) in build_search_space().items():
        search = GridSearchCV(make_pipe(estimator), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"{name:<20} best CV AUC = {search.best_score_:.3f}  {best_params[name]}")

    base = list(models.items())

    # 2) Ensembles built from the tuned base models
    models["Voting (soft)"] = VotingClassifier(estimators=base, voting="soft").fit(X_train, y_train)
    models["Stacking"] = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv,
    ).fit(X_train, y_train)

    # 3) Evaluate everything on the untouched test set
    rows, rocs, cms = [], {}, {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        m, roc, cm = evaluate(name, model, X_test, y_test, cv_auc)
        rows.append(m)
        rocs[name], cms[name] = roc, cm

    metrics = pd.DataFrame(rows).round(3)
    print("\nTest-set results\n", metrics.to_string(index=False))

    # 4) Feature importance from the Random Forest (for the app)
    rf = models["Random Forest"].named_steps["model"]
    importance = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

    # 5) Save artefacts
    joblib.dump(models, OUT_DIR / "models.joblib")
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    (OUT_DIR / "roc.json").write_text(json.dumps(rocs))
    (OUT_DIR / "confusion.json").write_text(json.dumps(cms))
    (OUT_DIR / "importance.json").write_text(json.dumps(importance))
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_params, default=str, indent=2))
    print(f"\nSaved models and metrics to {OUT_DIR}/")


if __name__ == "__main__":
    main()
