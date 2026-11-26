# eas510-project2-50740289
# Early-Warning Insider Threat Detection (CMU CERT r4.2)

**Author:** Radhika Utpal Yagnik (UB Person #50740289)  
**Course:** EAS 510: The Basics of AI — Fall 2026 (Project 2)  
**Official Final Report:** [`report.md`](report.md)

---

## Overview

This project investigates whether supervised machine learning models trained on daily enterprise behavioral logs can warn a Security Operations Center (SOC) **1 to 7 days *before* an insider initiates data theft**, while keeping false alarms within realistic daily analyst budgets ($\le 1$, $\le 3$, and $\le 5$ false positives per day). All post-attack-start insider rows are removed prior to splitting, and all models are evaluated under a strict chronological 60% / 20% / 20% split with zero temporal leakage.

### Key Results Summary (Full CMU CERT r4.2 — 328,638 User-Days)

* **Best Ranking Model (`Core + HTTP`, 39 behavioral features):** `HistGradientBoosting` achieved **0.1185 Test PR-AUC** (~269× above the 0.00044 random baseline), **0.1714 F1**, and **0.3750 precision** at **0.05 FP/day** (see [`results/full/comparison_table.csv`](results/full/comparison_table.csv)).
* **SOC Alert-Budget Performance ($\le 3.0$ FP/day):** Both `Random Forest` (0.57 actual FP/day) and `HistGradientBoosting` (1.08 actual FP/day) warned **3 of 7 unseen test insiders** early—with up to **6–7 days of lead time** on Scenario 2—cutting alert fatigue by 90%–95% compared to the static after-hours USB rule (2/7 detected at 10.42 FP/day; see [`results/full/soc_budget_table.csv`](results/full/soc_budget_table.csv)).
* **Key Negative Finding (Feature Ablation):** Adding static `psychometric.csv` (OCEAN) traits inflated Validation PR-AUC (0.1287 → 0.1546) while cutting `HistGradientBoosting` Test PR-AUC nearly in half (0.1185 → 0.0619) due to user-ID memorization (see [`results/full/ablation_table.csv`](results/full/ablation_table.csv)).

---

## Repository Structure

```text
eas510-project2-50740289/
├── README.md               # Setup, reproduction commands, and expected outputs
├── report.md               # Official final report
├── main.py                 # Single entry point (sample and --full modes)
├── config.py               # Seeds, split ratios, URL patterns, and hyperparameters
├── requirements.txt        # Pinned Python dependencies
├── src/
│   ├── data_prep.py        # Chunked log aggregation, rolling/baseline features, splits
│   ├── models.py           # Model pipelines, val-only thresholding, lead-time metrics
│   └── generate_sample.py  # Generates the 100% synthetic sample in data/sample/
├── data/
│   └── sample/             # Bundled synthetic sample for end-to-end verification
├── results/
│   ├── sample/             # Verified sample-run tables, JSON, and figures
│   └── full/               # Full CERT r4.2 report tables, JSON, and figures
└── .gitignore