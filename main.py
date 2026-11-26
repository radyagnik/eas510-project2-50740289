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
from sklearn.metrics import average_precision_score, precision_recall_curve

import config
from src.data_prep import build_daily_dataset, chronological_split
from src.models import (
    compute_scenario_breakdown,
    evaluate_predictions,
    fit_model,
    get_models,
    tune_threshold_f1,
    tune_threshold_fp_budget,
)


def split_feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    """Group engineered features into three ablation tiers."""
    psych_cols = {"O", "C", "E", "A", "N"}
    core = [c for c in all_features if "http" not in c and c not in psych_cols]
    core_http = [c for c in all_features if c not in psych_cols]
    return {
        "Core (4 logs)": core,
        "Core + HTTP": core_http,
        "Core + HTTP + Psych": list(all_features),
    }


def _plot_ablation(ablation_df: pd.DataFrame, out_path: Path, mode_label: str) -> None:
    pivot = ablation_df.pivot(index="model", columns="feature_set", values="test_pr_auc")
    ax = pivot.plot(kind="bar", figsize=(8, 5), width=0.75)
    ax.set_ylabel("Test PR-AUC (Higher is Better)")
    ax.set_xlabel("")
    ax.set_title(f"Feature Ablation Study — {mode_label}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Feature Set", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pr_curves(pr_curves: dict, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.5, 5))
    for name, ((prec, rec, _), ap) in pr_curves.items():
        plt.plot(rec, prec, linewidth=2, label=f"{name} (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_soc_budgets(budget_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.5, 5))
    for name, grp in budget_df.groupby("model"):
        grp = grp.sort_values("fp_budget")
        plt.plot(
            grp["fp_budget"],
            grp["test_insiders_detected"],
            marker="o",
            linewidth=2,
            label=name,
        )
    total_insiders = int(budget_df["test_insiders_total"].iloc[0])
    plt.xticks(config.SOC_FP_BUDGETS, [f"<= {b:g} FP/day" for b in config.SOC_FP_BUDGETS])
    plt.yticks(range(0, total_insiders + 1))
    plt.xlabel("Validation-Tuned SOC Alert Budget")
    plt.ylabel(f"Test Insiders Detected (out of {total_insiders})")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_feature_importance(imp_df: pd.DataFrame, out_path: Path, title: str) -> None:
    top = imp_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 5.5))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Random Forest Feature Importance (Gini)")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
            "For --full mode, extract CERT r4.2 and insiders.csv into data/raw/ or pass --data-dir."
        )

    print(f"=== Running Early-Warning Pipeline [{mode_label}] ===")
    df, insiders, all_features = build_daily_dataset(data_dir, use_cache=full_mode, rebuild_cache=rebuild_cache)
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

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # ------------------------------------------------------------------
    # Experiment 1: Controlled Feature Ablation
    # ------------------------------------------------------------------
    feature_sets = split_feature_sets(all_features)
    ablation_rows = []
    cached_outputs = {}  # (fs_name, model_name) -> (pipe, val_probs, test_probs)

    for fs_name, cols in feature_sets.items():
        X_tr, X_va, X_te = train_df[cols], val_df[cols], test_df[cols]
        for model_name, pipe in get_models().items():
            pipe = fit_model(model_name, pipe, X_tr, y_train)
            val_probs = pipe.predict_proba(X_va)[:, 1]
            test_probs = pipe.predict_proba(X_te)[:, 1]
            val_ap = float(average_precision_score(y_val, val_probs))
            test_ap = float(average_precision_score(y_test, test_probs))

            ablation_rows.append({
                "feature_set": fs_name,
                "n_features": len(cols),
                "model": model_name,
                "val_pr_auc": round(val_ap, 4),
                "test_pr_auc": round(test_ap, 4),
            })
            cached_outputs[(fs_name, model_name)] = (pipe, val_probs, test_probs)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "ablation_table.csv", index=False)
    _plot_ablation(ablation_df, out_dir / "ablation_pr_auc.png", mode_label)

    # Select best feature set strictly by mean Validation PR-AUC (zero test leakage)
    # (Static psychometric traits are retained in Experiment 1 ablation to document user-ID overfitting)
    behavioral_df = ablation_df[ablation_df["feature_set"] != "Core + HTTP + Psych"]
    val_summary = behavioral_df.groupby("feature_set")["val_pr_auc"].mean()
    best_fs = str(val_summary.idxmax())
    best_cols = feature_sets[best_fs]

    print("\n=== Experiment 1: Feature Ablation (Val & Test PR-AUC) ===")
    print(ablation_df.to_string(index=False))
    print(f"\nSelected best feature set on Validation mean PR-AUC: '{best_fs}' ({len(best_cols)} features)")

    # ------------------------------------------------------------------
    # Experiment 2: Primary Model Comparison (Validation-Tuned F1)
    # ------------------------------------------------------------------
    f1_results = []
    pr_curves = {}

    rule_preds = (test_df["after_hours_usb_count"] > 0).astype(int).to_numpy()
    f1_results.append(evaluate_predictions("Rule: After-Hours USB", test_df, rule_preds, None, threshold=1.0))

    for model_name in get_models().keys():
        pipe, val_probs, test_probs = cached_outputs[(best_fs, model_name)]
        best_t = tune_threshold_f1(y_val, val_probs)
        test_preds = (test_probs >= best_t).astype(int)
        f1_results.append(evaluate_predictions(model_name, test_df, test_preds, test_probs, best_t))
        pr_curves[model_name] = (
            precision_recall_curve(y_test, test_probs),
            average_precision_score(y_test, test_probs),
        )

    comparison_df = pd.DataFrame(f1_results)
    comparison_df.to_csv(out_dir / "comparison_table.csv", index=False)
    _plot_pr_curves(
        pr_curves,
        out_dir / "pr_curves.png",
        f"Precision-Recall Curves ({best_fs}) — {mode_label}",
    )

    print(f"\n=== Experiment 2: Test Set Comparison [{best_fs}, Val-Tuned F1] ===")
    print(comparison_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Experiment 3: SOC Alert-Budget Thresholding (<= 1, 3, 5 FP/day)
    # ------------------------------------------------------------------
    budget_rows = []
    scenario_rows = []

    for budget in config.SOC_FP_BUDGETS:
        for model_name in get_models().keys():
            _, val_probs, test_probs = cached_outputs[(best_fs, model_name)]
            t_budget = tune_threshold_fp_budget(val_df, val_probs, max_fp_per_day=budget)
            test_preds = (test_probs >= t_budget).astype(int)
            row = evaluate_predictions(
                model_name,
                test_df,
                test_preds,
                test_probs,
                t_budget,
                extra={"fp_budget": budget},
            )
            budget_rows.append(row)

            # Save scenario breakdown at the middle SOC budget (<= 3.0 FP/day)
            if budget == 3.0:
                scenario_rows.extend(compute_scenario_breakdown(model_name, test_df, test_preds))

    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(out_dir / "soc_budget_table.csv", index=False)
    _plot_soc_budgets(
        budget_df,
        out_dir / "soc_budget_tradeoff.png",
        f"Insider Detection Under SOC Alert Budgets ({best_fs}) — {mode_label}",
    )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "scenario_breakdown.csv", index=False)

    print(f"\n=== Experiment 3: SOC Alert-Budget Evaluation [{best_fs}] ===")
    cols_show = [
        "fp_budget", "model", "threshold", "precision", "recall",
        "false_positives_per_day", "test_insiders_detected",
        "insiders_warned_ge_3_days", "median_lead_time_days",
    ]
    print(budget_df[cols_show].to_string(index=False))

    print(f"\n=== Scenario Breakdown at <= 3.0 FP/day SOC Budget [{best_fs}] ===")
    print(scenario_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Experiment 4: Feature Importance (Random Forest on Selected Set)
    # ------------------------------------------------------------------
    rf_pipe, _, _ = cached_outputs[(best_fs, "Random Forest")]
    rf_imp = pd.DataFrame({
        "feature": best_cols,
        "importance": rf_pipe.named_steps["clf"].feature_importances_,
    }).sort_values("importance", ascending=False)
    rf_imp.to_csv(out_dir / "feature_importance.csv", index=False)
    _plot_feature_importance(
        rf_imp,
        out_dir / "feature_importance.png",
        f"Top 15 Random Forest Features ({best_fs}) — {mode_label}",
    )

    # Save master JSON artifact
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode_label,
                "horizon_days": config.PREDICTION_HORIZON_DAYS,
                "selected_feature_set": best_fs,
                "n_selected_features": len(best_cols),
                "ablation": ablation_rows,
                "comparison_f1": f1_results,
                "soc_budgets": budget_rows,
                "scenario_breakdown_3fp": scenario_rows,
            },
            f,
            indent=2,
        )

    print(f"\nSaved all tables, JSON, and 4 figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run early-warning insider threat pipeline.")
    parser.add_argument("--full", action="store_true", help="Run on full CERT r4.2 dataset.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional path to raw CERT r4.2 folder.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore daily_base_cache.csv and re-parse raw logs.")
    args = parser.parse_args()
    run_pipeline(full_mode=args.full, data_dir_override=args.data_dir, rebuild_cache=args.rebuild_cache)