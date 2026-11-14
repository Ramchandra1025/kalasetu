"""Leakage-safe demand data preparation.

The encoder is fitted on training rows only. The current dataset is synthetic,
so the resulting model is for pipeline validation, not market forecasting claims.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import (
    DATASET_FILE, RESULTS_DIR, FEATURE_COLUMNS, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, TARGET_COLUMN, VALIDATION_RATIO, TEST_RATIO,
    RANDOM_SEED,
)


def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Demand dataset not found: {DATASET_FILE}")
    return pd.read_csv(DATASET_FILE)


def validate_columns(dataframe):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def clean_dataset(dataframe):
    frame = dataframe.copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("unknown").astype(str).str.strip().str.lower()
        frame.loc[frame[column].eq(""), column] = "unknown"
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=NUMERIC_FEATURES + [TARGET_COLUMN])
    frame = frame[
        (frame[TARGET_COLUMN] >= 0)
        & (frame["historical_demand"] >= 0)
        & (frame["previous_demand"] >= 0)
        & (frame["price"] >= 0)
        & (frame["month"].between(1, 12))
    ]
    return frame.reset_index(drop=True)


def create_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _transform(frame, encoder):
    numeric = frame[NUMERIC_FEATURES].astype(float).to_numpy(dtype=np.float32)
    categorical = encoder.transform(frame[CATEGORICAL_FEATURES])
    encoded = np.asarray(categorical, dtype=np.float32)
    return np.concatenate([numeric, encoded], axis=1)


def save_metadata(encoder, feature_names, split_sizes):
    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target": TARGET_COLUMN,
        "encoder": "OneHotEncoder",
        "encoder_fit_scope": "training_rows_only",
        "handle_unknown": "ignore",
        "split_sizes": split_sizes,
        "data_type": "synthetic_pipeline_testing_data",
        "research_valid": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def create_datasets():
    dataframe = load_dataset()
    validate_columns(dataframe)
    dataframe = clean_dataset(dataframe)
    if len(dataframe) < 20:
        raise ValueError("Dataset must contain at least 20 valid rows.")

    # Split raw rows BEFORE fitting categorical preprocessing to avoid leakage.
    train_frame, temp_frame = train_test_split(
        dataframe, test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
    )
    validation_fraction = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    validation_frame, test_frame = train_test_split(
        temp_frame, test_size=1 - validation_fraction,
        random_state=RANDOM_SEED,
    )

    encoder = create_encoder()
    encoder.fit(train_frame[CATEGORICAL_FEATURES])
    feature_names = NUMERIC_FEATURES + encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()

    X_train = _transform(train_frame, encoder)
    X_validation = _transform(validation_frame, encoder)
    X_test = _transform(test_frame, encoder)
    y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_validation = validation_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

    split_sizes = {
        "train": len(train_frame),
        "validation": len(validation_frame),
        "test": len(test_frame),
    }
    save_metadata(encoder, feature_names, split_sizes)

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "encoder": encoder,
        "feature_names": feature_names,
        "dataframe": dataframe,
        "split_sizes": split_sizes,
    }


if __name__ == "__main__":
    data = create_datasets()
    for key in ("X_train", "X_validation", "X_test"):
        print(f"{key}: {data[key].shape}")
    print(f"Split sizes: {data['split_sizes']}")
