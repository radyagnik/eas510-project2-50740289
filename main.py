"""Single entry point for EAS 510 Project 2: Early-Warning Insider Threat Detection.

Usage:
    python main.py          # Run on bundled synthetic sample -> results/sample/
    python main.py --full   # Run on local CERT r4.2 in data/raw/ -> results/full/
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_prep import build_daily_dataset, chronological_split


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Pick probability threshold on validation set that maximizes F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def compute_lead_time(test_df: pd.DataFrame, preds: np.ndarray) -> dict:
    """Measure how many days before attack start each test insider is first warned."""
    eval_df = test_df[["user", "day", "attack_start_day", "label"]].copy()
    eval_df["pred"] = preds

    # Test insiders are users who have at least one positive pre-theft day in test_df
    test_insiders = eval_df.loc[eval_df["label"] == 1, "user"].unique()
    lead_times = []

    for user in test_insiders:
        u_df = eval_df[(eval_df["user"] == user) & (eval_df["label"] == 1)]
        flagged = u_df[u_df["pred"] == 1]
        if not flagged.empty:
            first_alert = flagged["day"].min()
            attack_start = flagged["attack_start_day"].iloc[0]
            lead_times.append(int((attack_start - first_alert).days))

    n_total = len(test_insiders)
    n_detected = len(lead_times)
    n_ge_3 = sum(lt >= 3 for lt in lead_times)

    return {
        "test_insiders_total": int(n_total),
        "test_insiders_detected": int(n_detected),
        "insiders_warned_ge_3_days": int(n_ge_3),
        "median_lead_time_days": float(np.median(lead_times)) if lead_times else 0.0,
        "mean_lead_time_days": float(np.mean(lead_times)) if lead_times else 0.0,
    }


def evaluate_predictions(name: str, test_df: pd.DataFrame, preds: np.ndarray, probs: np.ndarray | None, threshold: float) -> dict:
    y_test = test_df["label"].to_numpy()
    n_test_days = max(1, test_df["day"].nunique())
    fp_count = int(((preds == 1) & (y_test == 0)).sum())

    metrics = {
        "model": name,
        "threshold": round(threshold, 3),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4) if probs is not None else None,
        "false_positives_per_day": round(fp_count / n_test_days, 2),
    }
    metrics.update(compute_lead_time(test_df, preds))
    return metrics


def run_pipeline(full_mode: bool, data_dir_override: str | None = None, rebuild_cache: bool = False) -> None:
    if data_dir_override:
        data_dir = Path(data_dir_override)
    else:
        data_dir = config.DATA_RAW_DIR if full_mode else config.DATA_SAMPLE_DIR

    
    out_dir = config.RESULTS_FULL_DIR if full_mode else config.RESULTS_SAMPLE_DIR
    mode_label = "FULL (CERT r4.2)" if full_mode else "SAMPLE (SYNTHETIC - pipeline check only)"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "logon.csv").exists():
        raise FileNotFoundError(
            f"Could not find logon.csv in {data_dir}. "
            "For --full mode, extract CERT r4.2 and insiders.csv into data/raw/."
        )

    print(f"=== Running Early-Warning Pipeline [{mode_label}] ===")
    df, insiders, feature_cols = build_daily_dataset(data_dir, use_cache=full_mode, rebuild_cache=rebuild_cache)
    train_df, val_df, test_df = chronological_split(df)

    print(f"Total user-days: {len(df):,} | Positive warning days: {int(df['label'].sum()):,}")
    print(
        f"Train: {len(train_df):,} rows ({train_df['day'].min().date()} to {train_df['day'].max().date()}), "
        f"pos={int(train_df['label'].sum())}"
    )
    print(
        f"Val:   {len(val_df):,} rows ({val_df['day'].min().date()} to {val_df['day'].max().date()}), "
        f"pos={int(val_df['label'].sum())}"
    )
    print(
        f"Test:  {len(test_df):,} rows ({test_df['day'].min().date()} to {test_df['day'].max().date()}), "
        f"pos={int(test_df['label'].sum())}"
    )

    X_train, y_train = train_df[feature_cols], train_df["label"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["label"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["label"].to_numpy()

    results = []
    pr_curves = {}

    # 1. Rule-based baseline: flag any day with after-hours USB use
    rule_preds = (test_df["after_hours_usb_count"] > 0).astype(int).to_numpy()
    results.append(evaluate_predictions("Rule: After-Hours USB", test_df, rule_preds, None, threshold=1.0))

    # 2. Supervised ML models
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=config.RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1)),
        ]),
    }

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        best_t = tune_threshold(y_val, val_probs)

        test_probs = pipe.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_t).astype(int)

        results.append(evaluate_predictions(name, test_df, test_preds, test_probs, best_t))
        pr_curves[name] = (precision_recall_curve(y_test, test_probs), average_precision_score(y_test, test_probs))

    results_df = pd.DataFrame(results)
    print("\n=== Test Set Comparison ===")
    print(results_df.to_string(index=False))

    # Save tables & JSON
    results_df.to_csv(out_dir / "comparison_table.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"mode": mode_label, "horizon_days": config.PREDICTION_HORIZON_DAYS, "results": results}, f, indent=2)

    # Save Precision-Recall Curve figure
    plt.figure(figsize=(7, 5))
    for name, ((prec, rec, _), ap) in pr_curves.items():
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves — {mode_label}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close()

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run early-warning insider threat pipeline.")
    parser.add_argument("--full", action="store_true", help="Run on full CERT r4.2 dataset.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional path to raw CERT r4.2 folder.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore daily_base_cache.csv and re-parse raw logs.")
    args = parser.parse_args()
    run_pipeline(full_mode=args.full, data_dir_override=args.data_dir, rebuild_cache=args.rebuild_cache)