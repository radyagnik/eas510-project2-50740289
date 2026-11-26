from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_SAMPLE_DIR = ROOT_DIR / "data" / "sample"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
RESULTS_SAMPLE_DIR = ROOT_DIR / "results" / "sample"
RESULTS_FULL_DIR = ROOT_DIR / "results" / "full"

# Reproducibility
RANDOM_SEED = 42

# Early-warning problem settings
PREDICTION_HORIZON_DAYS = 7   # Predict whether theft starts within the next 7 days
WORK_START_HOUR = 7           # After-hours = before 7:00 AM or at/after 7:00 PM
WORK_END_HOUR = 19
ROLLING_WINDOW_DAYS = 7
COMPANY_DOMAIN = "@dtaa.com"

# Chronological split ratios (by date: earliest -> latest)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Chunk size for reading large CERT CSVs safely in low RAM
CHUNK_SIZE = 200_000

# Regex patterns for high-signal web domains in http.csv (case-insensitive)
JOB_SEARCH_URL_PATTERN = r"indeed|monster|linkedin|careerbuilder|simplyhired|glassdoor|job|career|resume"
UPLOAD_CLOUD_URL_PATTERN = r"wikileaks|dropbox|drive\.google|upload|sendspace|mediafire|rapidshare|megaupload|pastebin|box\.com"

# SOC alert-budget thresholds (max false positives per day, tuned on validation set)
SOC_FP_BUDGETS = [1.0, 3.0, 5.0]

# Model hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 5
HGB_MAX_ITER = 200
HGB_MAX_DEPTH = 6
HGB_LEARNING_RATE = 0.05