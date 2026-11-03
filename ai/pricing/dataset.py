"""
KalaSetu - Fair Pricing AI
Dataset Processing

Responsibilities:
    1. Load pricing dataset
    2. Validate columns
    3. Clean data
    4. Encode categorical features
    5. Split into train/validation/test
    6. Return processed datasets
"""

import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import (
    DATASET_FILE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    RESULTS_DIR,
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN]
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"\nPricing dataset not found:\n"
            f"{DATASET_FILE}\n\n"
            f"Create pricing.csv before running "
            f"the dataset pipeline."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    print(
        f"Dataset loaded: "
        f"{len(dataframe)} rows"
    )

    return dataframe


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(dataframe):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print("Required columns validated.")


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(dataframe):

    dataframe = dataframe.copy()

    # Remove completely empty rows
    dataframe = dataframe.dropna(
        how="all"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        dataframe[column] = dataframe[column].replace(
            {
                "nan": np.nan,
                "none": np.nan,
                "": np.nan,
            }
        )

    # --------------------------------------------------------
    # Remove rows with missing required values
    # --------------------------------------------------------

    before = len(dataframe)

    dataframe = dataframe.dropna(
        subset=REQUIRED_COLUMNS
    )

    removed = before - len(dataframe)

    print(
        f"Removed invalid rows: {removed}"
    )

    # --------------------------------------------------------
    # Remove impossible values
    # --------------------------------------------------------

    dataframe = dataframe[
        (dataframe["size_cm"] > 0)
        & (dataframe["production_hours"] > 0)
        & (dataframe["material_cost"] >= 0)
        & (dataframe["labor_cost"] >= 0)
        & (dataframe["demand_score"] >= 0)
        & (dataframe["demand_score"] <= 100)
        & (dataframe["complexity_score"] >= 0)
        & (dataframe["complexity_score"] <= 100)
        & (dataframe[TARGET_COLUMN] > 0)
    ]

    dataframe = dataframe.reset_index(
        drop=True
    )

    print(
        f"Clean dataset size: "
        f"{len(dataframe)} rows"
    )

    return dataframe


# ============================================================
# FEATURE ENCODER
# ============================================================

def create_encoder():

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        # Compatibility with older scikit-learn versions
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    return encoder


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_features(dataframe):

    numeric_data = dataframe[
        NUMERIC_FEATURES
    ].astype(float)

    categorical_data = dataframe[
        CATEGORICAL_FEATURES
    ].astype(str)

    encoder = create_encoder()

    encoded_categories = encoder.fit_transform(
        categorical_data
    )

    encoded_category_names = (
        encoder
        .get_feature_names_out(
            CATEGORICAL_FEATURES
        )
        .tolist()
    )

    numeric_array = numeric_data.to_numpy(
        dtype=np.float32
    )

    encoded_array = np.asarray(
        encoded_categories,
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            numeric_array,
            encoded_array,
        ],
        axis=1,
    )

    target = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    feature_names = (
        NUMERIC_FEATURES
        + encoded_category_names
    )

    return (
        features,
        target,
        encoder,
        feature_names,
    )


# ============================================================
# DATA SPLIT
# ============================================================

def split_dataset(features, target):

    # --------------------------------------------------------
    # First split:
    # Train = 70%
    # Temporary = 30%
    # --------------------------------------------------------

    validation_test_ratio = (
        VALIDATION_RATIO
        + TEST_RATIO
    )

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            features,
            target,
            test_size=validation_test_ratio,
            random_state=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Second split:
    # Validation = 15%
    # Test = 15%
    #
    # Since temporary = 30%,
    # each receives 50% of temporary.
    # --------------------------------------------------------

    test_ratio_inside_temp = (
        TEST_RATIO
        / validation_test_ratio
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=test_ratio_inside_temp,
            random_state=RANDOM_SEED,
        )
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
# SAVE PREPROCESSING METADATA
# ============================================================

def save_metadata(
    encoder,
    feature_names,
):

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "feature_names": feature_names,
        "categories": [
            categories.tolist()
            for categories
            in encoder.categories_
        ],
        "random_seed": RANDOM_SEED,
    }

    metadata_file = (
        RESULTS_DIR
        / "preprocessing_metadata.json"
    )

    with open(
        metadata_file,
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
        f"{metadata_file}"
    )


# ============================================================
# COMPLETE DATA PIPELINE
# ============================================================

def create_datasets():

    print("=" * 60)
    print("KALASETU - FAIR PRICING DATASET")
    print("=" * 60)

    # Load
    dataframe = load_dataset()

    # Validate
    validate_columns(
        dataframe
    )

    # Clean
    dataframe = clean_dataset(
        dataframe
    )

    if len(dataframe) < 10:

        raise ValueError(
            "\nDataset contains fewer than "
            "10 valid rows.\n"
            "Add more data before training."
        )

    # Prepare
    (
        features,
        target,
        encoder,
        feature_names,
    ) = prepare_features(
        dataframe
    )

    print(
        f"\nTotal features after encoding: "
        f"{features.shape[1]}"
    )

    # Split
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

    print("\nDataset Split:")
    print(
        f"  Train      : {len(X_train)}"
    )
    print(
        f"  Validation : {len(X_validation)}"
    )
    print(
        f"  Test       : {len(X_test)}"
    )

    # Save metadata
    save_metadata(
        encoder,
        feature_names,
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

    try:

        datasets = create_datasets()

        print("\n" + "=" * 60)
        print("DATASET PIPELINE TEST PASSED")
        print("=" * 60)

    except Exception as error:

        print("\n" + "=" * 60)
        print("DATASET PIPELINE TEST FAILED")
        print("=" * 60)

        print(f"\nError: {error}")