"""
KalaSetu - Demand Prediction AI
Dataset Loading and Preprocessing
"""

import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import (
    DATASET_FILE,
    RESULTS_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Demand dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(DATASET_FILE)

    print(
        f"Dataset loaded: {len(dataframe)} rows"
    )

    return dataframe


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(dataframe):

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print("Required columns validated.")


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(dataframe):

    dataframe = dataframe.copy()

    # Numeric columns
    for column in (
        NUMERIC_FEATURES
        + [TARGET_COLUMN]
    ):
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # Categorical columns
    for column in CATEGORICAL_FEATURES:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    before_count = len(dataframe)

    dataframe = dataframe.dropna(
        subset=FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    # Valid demand
    dataframe = dataframe[
        dataframe[TARGET_COLUMN] >= 0
    ]

    dataframe = dataframe[
        dataframe["historical_demand"] >= 0
    ]

    dataframe = dataframe[
        dataframe["previous_demand"] >= 0
    ]

    dataframe = dataframe[
        dataframe["price"] >= 0
    ]

    dataframe = dataframe[
        dataframe["month"].between(1, 12)
    ]

    after_count = len(dataframe)

    print(
        f"Removed invalid rows: "
        f"{before_count - after_count}"
    )

    print(
        f"Clean dataset size: "
        f"{after_count} rows"
    )

    return dataframe.reset_index(drop=True)


# ============================================================
# ENCODER
# ============================================================

def create_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(dataframe):

    numeric_data = dataframe[
        NUMERIC_FEATURES
    ].astype(float)

    categorical_data = dataframe[
        CATEGORICAL_FEATURES
    ]

    encoder = create_encoder()

    encoded_categories = encoder.fit_transform(
        categorical_data
    )

    encoded_feature_names = (
        encoder
        .get_feature_names_out(
            CATEGORICAL_FEATURES
        )
        .tolist()
    )

    feature_names = (
        NUMERIC_FEATURES
        + encoded_feature_names
    )

    feature_matrix = np.concatenate(
        [
            numeric_data.to_numpy(
                dtype=np.float32
            ),
            np.asarray(
                encoded_categories,
                dtype=np.float32,
            ),
        ],
        axis=1,
    )

    target = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    print(
        f"Total features after encoding: "
        f"{feature_matrix.shape[1]}"
    )

    return (
        feature_matrix,
        target,
        encoder,
        feature_names,
    )


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(
    features,
    target,
):

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        features,
        target,
        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),
        random_state=RANDOM_SEED,
    )

    validation_fraction = (
        VALIDATION_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=(
            1 - validation_fraction
        ),
        random_state=RANDOM_SEED,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    encoder,
    feature_names,
):

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target": TARGET_COLUMN,
        "encoder": "OneHotEncoder",
        "handle_unknown": "ignore",
        "data_type": (
            "synthetic_pipeline_testing_data"
        ),
        "research_valid": False,
    }

    output_file = (
        RESULTS_DIR
        / "preprocessing_metadata.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"Preprocessing metadata saved:\n"
        f"{output_file}"
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def create_datasets():

    dataframe = load_dataset()

    validate_columns(
        dataframe
    )

    dataframe = clean_dataset(
        dataframe
    )

    if len(dataframe) < 20:

        raise ValueError(
            "Dataset must contain at least "
            "20 valid rows."
        )

    (
        features,
        target,
        encoder,
        feature_names,
    ) = prepare_features(
        dataframe
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(
        features,
        target,
    )

    save_metadata(
        encoder,
        feature_names,
    )

    print(
        f"Train samples      : {len(X_train)}"
    )

    print(
        f"Validation samples : "
        f"{len(X_validation)}"
    )

    print(
        f"Test samples       : {len(X_test)}"
    )

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
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("KALASETU - DEMAND DATASET TEST")
    print("=" * 70)

    datasets = create_datasets()

    print("\nDataset Test Summary")

    print(
        f"Training shape     : "
        f"{datasets['X_train'].shape}"
    )

    print(
        f"Validation shape   : "
        f"{datasets['X_validation'].shape}"
    )

    print(
        f"Test shape         : "
        f"{datasets['X_test'].shape}"
    )

    print(
        f"Feature count      : "
        f"{len(datasets['feature_names'])}"
    )

    print(
        "\nDATASET PIPELINE TEST PASSED."
    )

    print("=" * 70)