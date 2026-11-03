"""
KalaSetu - Fair Pricing AI
Configuration

Model:
    XGBoost Regressor

Task:
    Predict a fair/recommended price for an artisan product.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "pricing"
)

DATASET_FILE = DATASET_ROOT / "pricing.csv"

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pricing"
)

RESEARCH_DIR = PROJECT_ROOT / "research"

EXPERIMENT_DIR = (
    RESEARCH_DIR
    / "experiments"
    / "pricing"
)

EVALUATION_DIR = (
    RESEARCH_DIR
    / "evaluation"
    / "pricing"
)

RESULTS_DIR = (
    RESEARCH_DIR
    / "results"
    / "pricing"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "xgboost_regressor"

MODEL_VERSION = "1.0.0"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "fair_price"


# ============================================================
# INPUT FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "size_cm",
    "production_hours",
    "material_cost",
    "labor_cost",
    "demand_score",
    "complexity_score",
]


CATEGORICAL_FEATURES = [
    "craft_category",
    "material",
    "region",
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

N_ESTIMATORS = 300

MAX_DEPTH = 6

LEARNING_RATE = 0.05

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

MIN_CHILD_WEIGHT = 1

REG_ALPHA = 0.0

REG_LAMBDA = 1.0


# ============================================================
# PRICE OUTPUT CONFIGURATION
# ============================================================

MIN_PREDICTED_PRICE = 1.0

ROUND_PRICE_TO = 1


# ============================================================
# DIRECTORY CREATION
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True)


# ============================================================
# CONFIGURATION TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("KALASETU - FAIR PRICING CONFIGURATION")
    print("=" * 60)

    print(f"Project Root       : {PROJECT_ROOT}")
    print(f"Dataset Root       : {DATASET_ROOT}")
    print(f"Dataset File       : {DATASET_FILE}")
    print(f"Model Directory    : {MODEL_DIR}")

    print("\nModel:")
    print(f"  Name             : {MODEL_NAME}")
    print(f"  Version          : {MODEL_VERSION}")

    print("\nTarget:")
    print(f"  Target Column    : {TARGET_COLUMN}")

    print("\nNumeric Features:")

    for feature in NUMERIC_FEATURES:
        print(f"  - {feature}")

    print("\nCategorical Features:")

    for feature in CATEGORICAL_FEATURES:
        print(f"  - {feature}")

    print("\nDataset Split:")
    print(f"  Train            : {TRAIN_RATIO}")
    print(f"  Validation       : {VALIDATION_RATIO}")
    print(f"  Test             : {TEST_RATIO}")

    print("\nXGBoost:")
    print(f"  Estimators       : {N_ESTIMATORS}")
    print(f"  Max Depth        : {MAX_DEPTH}")
    print(f"  Learning Rate    : {LEARNING_RATE}")
    print(f"  Subsample        : {SUBSAMPLE}")
    print(f"  Colsample        : {COLSAMPLE_BYTREE}")

    print("\nConfiguration test PASSED.")