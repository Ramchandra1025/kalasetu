"""
KalaSetu - Fair Pricing AI
Training Pipeline

Purpose:
    Train the XGBoost fair-price regression model.

IMPORTANT:
    The current pricing dataset is synthetic and is used only
    to validate the complete ML pipeline.

    These results MUST NOT be presented as research findings.
"""

import json
from datetime import datetime

from .dataset import create_datasets
from .model import (
    train_model,
    predict_price,
    save_model,
    get_feature_importance,
)

from .config import (
    DATASET_FILE,
    EXPERIMENT_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)


def main():

    print("=" * 70)
    print("KALASETU - FAIR PRICING AI TRAINING")
    print("=" * 70)

    # ============================================================
    # 1. CREATE DATASETS
    # ============================================================

    print("\n[1/5] Preparing pricing dataset...")

    datasets = create_datasets()

    X_train = datasets["X_train"]
    X_validation = datasets["X_validation"]
    X_test = datasets["X_test"]

    y_train = datasets["y_train"]
    y_validation = datasets["y_validation"]
    y_test = datasets["y_test"]

    encoder = datasets["encoder"]
    feature_names = datasets["feature_names"]

    print("\nDataset preparation completed.")

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_validation)}")
    print(f"  Test samples       : {len(X_test)}")
    print(f"  Total features     : {len(feature_names)}")

    # ============================================================
    # 2. TRAIN MODEL
    # ============================================================

    print("\n[2/5] Training XGBoost regression model...")

    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print("Training completed successfully.")

    # ============================================================
    # 3. VALIDATION PREDICTIONS
    # ============================================================

    print("\n[3/5] Generating validation predictions...")

    validation_predictions = predict_price(
        model,
        X_validation,
    )

    print(
        f"Validation predictions generated: "
        f"{len(validation_predictions)}"
    )

    print("\nValidation Predictions:")

    for index, prediction in enumerate(validation_predictions):

        actual = float(y_validation[index])

        print(
            f"  Sample {index + 1}: "
            f"Actual = ₹{actual:.2f} | "
            f"Predicted = ₹{prediction:.2f}"
        )

    # ============================================================
    # 4. FEATURE IMPORTANCE
    # ============================================================

    print("\n[4/5] Calculating feature importance...")

    feature_importance = get_feature_importance(
        model,
        feature_names,
    )

    print("\nTop Feature Importance:")

    for item in feature_importance[:10]:

        print(
            f"  {item['feature']:<30} "
            f"{item['importance']:.6f}"
        )

    # ============================================================
    # 5. SAVE MODEL + EXPERIMENT
    # ============================================================

    print("\n[5/5] Saving trained model...")

    save_model(
        model,
        encoder,
        feature_names,
    )

    # ============================================================
    # TRAINING SUMMARY
    # ============================================================

    experiment_summary = {
        "experiment_name": "fair_pricing_xgboost_pipeline_test",
        "model_name": "xgboost_regressor",
        "model_version": "1.0.0",

        "dataset_file": str(DATASET_FILE),

        "dataset_size": int(
            len(X_train)
            + len(X_validation)
            + len(X_test)
        ),

        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_validation)),
        "test_samples": int(len(X_test)),

        "feature_count": int(len(feature_names)),

        "random_seed": RANDOM_SEED,

        "data_type": "synthetic_pipeline_testing_data",

        "research_valid": False,

        "warning": (
            "The current dataset is synthetic and is used "
            "only for pipeline validation. These results "
            "must not be presented as research findings."
        ),

        "created_at": datetime.now().isoformat(),

        "validation_predictions": [
            {
                "actual_price": float(
                    y_validation[index]
                ),
                "predicted_price": float(
                    validation_predictions[index]
                ),
            }
            for index in range(
                len(validation_predictions)
            )
        ],

        "feature_importance": feature_importance,
    }

    summary_file = (
        EXPERIMENT_DIR
        / "training_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_summary,
            file,
            indent=4,
        )

    # ============================================================
    # TRAINING RESULT
    # ============================================================

    training_result = {
        "model": "XGBRegressor",

        "task": "fair_price_regression",

        "dataset_size": int(
            len(X_train)
            + len(X_validation)
            + len(X_test)
        ),

        "train_size": int(len(X_train)),
        "validation_size": int(len(X_validation)),
        "test_size": int(len(X_test)),

        "feature_count": int(len(feature_names)),

        "research_valid": False,

        "status": "training_completed",
    }

    result_file = (
        RESULTS_DIR
        / "training_result.json"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            training_result,
            file,
            indent=4,
        )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print("\nModel:")
    print("  XGBoost Regressor")

    print("\nDataset:")
    print(f"  {DATASET_FILE}")

    print("\nSamples:")
    print(f"  Train      : {len(X_train)}")
    print(f"  Validation : {len(X_validation)}")
    print(f"  Test       : {len(X_test)}")

    print("\nFiles generated:")

    print(
        "  Model      : "
        "ml/models/pricing/fair_pricing_model.json"
    )

    print(
        "  Encoder    : "
        "ml/models/pricing/feature_encoder.joblib"
    )

    print(
        "  Metadata   : "
        "ml/models/pricing/model_metadata.json"
    )

    print(
        "  Experiment : "
        "research/experiments/pricing/"
        "training_summary.json"
    )

    print(
        "  Result     : "
        "research/results/pricing/"
        "training_result.json"
    )

    print("\nIMPORTANT:")
    print(
        "  Current dataset is synthetic "
        "pipeline-testing data."
    )

    print(
        "  Research-valid result: NO"
    )

    print("\nTRAINING PIPELINE TEST PASSED.")


if __name__ == "__main__":
    main()