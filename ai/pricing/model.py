"""
KalaSetu - Fair Pricing AI
XGBoost Regression Model
"""

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from .config import (
    MODEL_DIR,
    MODEL_NAME,
    MODEL_VERSION,
    N_ESTIMATORS,
    MAX_DEPTH,
    LEARNING_RATE,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    MIN_CHILD_WEIGHT,
    REG_ALPHA,
    REG_LAMBDA,
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_FILE = MODEL_DIR / "fair_pricing_model.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
ENCODER_FILE = MODEL_DIR / "feature_encoder.joblib"


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():
    """
    Create the XGBoost regression model.
    """

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=42,
        n_jobs=-1,
    )

    return model


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
):
    """
    Train the XGBoost pricing model.
    """

    model = create_model()

    eval_set = None

    if (
        X_validation is not None
        and y_validation is not None
    ):

        eval_set = [
            (X_train, y_train),
            (X_validation, y_validation),
        ]

    if eval_set:

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

    else:

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_price(
    model,
    features,
):
    """
    Predict fair price.

    Ensures:
        - no negative price
        - finite output
    """

    predictions = model.predict(
        features
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    predictions = np.maximum(
        predictions,
        1.0,
    )

    return predictions


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names,
):
    """
    Return feature importance values.
    """

    importance = model.feature_importances_

    result = []

    for name, value in zip(
        feature_names,
        importance,
    ):

        result.append(
            {
                "feature": name,
                "importance": float(value),
            }
        )

    result.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return result


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    encoder,
    feature_names,
):
    """
    Save trained model and preprocessing objects.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # XGBoost model
    # --------------------------------------------------------

    model.save_model(
        str(MODEL_FILE)
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    joblib.dump(
        encoder,
        ENCODER_FILE,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "XGBRegressor",
        "target": "fair_price",
        "feature_count": len(
            feature_names
        ),
        "feature_names": feature_names,
        "model_file": MODEL_FILE.name,
        "encoder_file": ENCODER_FILE.name,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"\nModel saved:"
        f"\n  {MODEL_FILE}"
    )

    print(
        f"Encoder saved:"
        f"\n  {ENCODER_FILE}"
    )

    print(
        f"Metadata saved:"
        f"\n  {METADATA_FILE}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained XGBoost model.
    """

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Pricing model not found:\n"
            f"{MODEL_FILE}"
        )

    model = xgb.XGBRegressor()

    model.load_model(
        str(MODEL_FILE)
    )

    return model


# ============================================================
# LOAD ENCODER
# ============================================================

def load_encoder():
    """
    Load saved feature encoder.
    """

    if not ENCODER_FILE.exists():

        raise FileNotFoundError(
            f"Feature encoder not found:\n"
            f"{ENCODER_FILE}"
        )

    return joblib.load(
        ENCODER_FILE
    )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("KALASETU - FAIR PRICING MODEL")
    print("=" * 60)

    print("\nCreating XGBoost model...")

    model = create_model()

    print(
        f"Model Type : "
        f"{type(model).__name__}"
    )

    print(
        f"Estimators : "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Max Depth  : "
        f"{MAX_DEPTH}"
    )

    print(
        f"Learning Rate : "
        f"{LEARNING_RATE}"
    )

    print(
        "\nModel creation test PASSED."
    )