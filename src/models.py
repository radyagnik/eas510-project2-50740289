"""Model definitions, validation-only threshold tuning, and early-warning evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config


def get_models() -> dict[str, Pipeline]:
    """Return all supervised ML pipelines configured from config.py."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=config.HGB_MAX_ITER,
                max_depth=config.HGB_MAX_DEPTH,
                learning_rate=config.HGB_LEARNING_RATE,
                min_samples_leaf=20,
                random_state=config.RANDOM_SEED,
            )),
        ]),
    }


def fit_model(name: str, pipe: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    """Fit model, applying balanced sample weights for HistGradientBoosting."""
    if name == "HistGradientBoosting":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        pipe.fit(X_train, y_train, clf__sample_weight=weights)
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    """Build a dense candidate threshold grid combining linear steps and high-probability quantiles."""
    lin = np.linspace(0.01, 0.99, 99)
    qs = np.quantile(probs, np.linspace(0.90, 0.9999, 100))
    return np.unique(np.clip(np.concatenate([lin, qs]), 0.0, 1.0))


def tune_threshold_f1(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Pick probability threshold on validation set that maximizes F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in _candidate_thresholds(val_probs):
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def tune_threshold_fp_budget(val_df: pd.DataFrame, val_probs: np.ndarray, max_fp_per_day: float) -> float:
    """Pick the lowest validation threshold whose validation FP/day <= max_fp_per_day.

    Lowering the threshold maximizes recall while respecting the SOC analyst alert budget.
    Strictly tuned on validation data (never touches test set).
    """
    y_val = val_df["label"].to_numpy()
    n_val_days = max(1, val_df["day"].nunique())
    candidates = np.sort(_candidate_thresholds(val_probs))

    for t in candidates:
        preds = (val_probs >= t).astype(int)
        fp_per_day = int(((preds == 1) & (y_val == 0)).sum()) / n_val_days
        if fp_per_day <= max_fp_per_day:
            return float(t)

    return float(candidates[-1])


def compute_lead_time(test_df: pd.DataFrame, preds: np.ndarray) -> dict:
    """Measure how many days before attack start each test insider is first warned."""
    eval_df = test_df[["user", "day", "attack_start_day", "label"]].copy()
    eval_df["pred"] = preds

    test_insiders = eval_df.loc[eval_df["label"] == 1, "user"].unique()
    lead_times = []

    for user in test_insiders:
        u_df = eval_df[(eval_df["user"] == user) & (eval_df["label"] == 1)]
        flagged = u_df[u_df["pred"] == 1]
        if not flagged.empty:
            first_alert = flagged["day"].min()
            attack_start = flagged["attack_start_day"].iloc[0]
            lead_times.append(int((attack_start - first_alert).days))

    return {
        "test_insiders_total": int(len(test_insiders)),
        "test_insiders_detected": int(len(lead_times)),
        "insiders_warned_ge_3_days": int(sum(lt >= 3 for lt in lead_times)),
        "median_lead_time_days": round(float(np.median(lead_times)), 2) if lead_times else 0.0,
        "mean_lead_time_days": round(float(np.mean(lead_times)), 2) if lead_times else 0.0,
    }


def evaluate_predictions(
    name: str,
    test_df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray | None,
    threshold: float,
    extra: dict | None = None,
) -> dict:
    y_test = test_df["label"].to_numpy()
    n_test_days = max(1, test_df["day"].nunique())
    fp_count = int(((preds == 1) & (y_test == 0)).sum())

    metrics = {
        "model": name,
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4) if probs is not None else None,
        "false_positives_per_day": round(fp_count / n_test_days, 2),
    }
    metrics.update(compute_lead_time(test_df, preds))
    if extra:
        metrics.update(extra)
    return metrics


def compute_scenario_breakdown(model_name: str, test_df: pd.DataFrame, preds: np.ndarray) -> list[dict]:
    """Break down insider detection and lead time by CERT scenario (1, 2, 3)."""
    eval_df = test_df[["user", "day", "scenario", "attack_start_day", "label"]].copy()
    eval_df["pred"] = preds
    pos_df = eval_df[eval_df["label"] == 1]

    rows = []
    for scenario, s_df in pos_df.groupby("scenario"):
        users = s_df["user"].unique()
        lead_times = []
        for user in users:
            u_df = s_df[s_df["user"] == user]
            flagged = u_df[u_df["pred"] == 1]
            if not flagged.empty:
                first_alert = flagged["day"].min()
                attack_start = flagged["attack_start_day"].iloc[0]
                lead_times.append(int((attack_start - first_alert).days))

        rows.append({
            "model": model_name,
            "scenario": int(scenario),
            "insiders_total": int(len(users)),
            "insiders_detected": int(len(lead_times)),
            "median_lead_time_days": round(float(np.median(lead_times)), 2) if lead_times else 0.0,
        })
    return rows