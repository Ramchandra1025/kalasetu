"""
KalaSetu - Demand Prediction AI
Training Pipeline
"""

import json

from .dataset import create_datasets

from .model import (
    train_model,
    predict_demand,
    get_feature_importance,
    save_model,
)

from .config import (
    EXPERIMENT_DIR,
    RESULTS_DIR,
)


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("KALASETU - DEMAND PREDICTION TRAINING")
    print("=" * 70)

    datasets = create_datasets()

    X_train = datasets["X_train"]
    X_validation = datasets[
        "X_validation"
    ]
    y_train = datasets["y_train"]
    y_validation = datasets[
        "y_validation"
    ]

    encoder = datasets["encoder"]

    feature_names = datasets[
        "feature_names"
    ]

    print("\nCreating XGBoost model...")

    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print(
        "\nTraining completed successfully."
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = (
        predict_demand(
            model,
            X_validation,
        )
    )

    print(
        "\nValidation Predictions:"
    )

    for index, (
        actual,
        predicted,
    ) in enumerate(
        zip(
            y_validation,
            validation_predictions,
        ),
        start=1,
    ):

        print(
            f"Sample {index} "
            f"Actual {actual:.2f} "
            f"Predicted {predicted:.2f}"
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        get_feature_importance(
            model,
            feature_names,
        )
    )

    print(
        "\nTop Feature Importance:"
    )

    for item in importance[:10]:

        print(
            f"{item['feature']:<35}"
            f"{item['importance']:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        encoder,
        feature_names,
    )

    print(
        "\nModel saved successfully."
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    summary = {

        "model": "XGBRegressor",

        "model_name":
            "xgboost_demand_regressor",

        "model_version":
            "1.0.0",

        "target":
            "future_demand",

        "train_samples":
            len(X_train),

        "validation_samples":
            len(X_validation),

        "feature_count":
            len(feature_names),

        "feature_importance":
            importance,

        "data_type":
            "synthetic_pipeline_testing_data",

        "research_valid":
            False,

        "research_note":
            (
                "Current demand dataset is "
                "synthetic pipeline-testing data. "
                "Metrics must not be presented "
                "as research-valid performance."
            ),
    }

    experiment_file = (
        EXPERIMENT_DIR
        / "training_summary.json"
    )

    result_file = (
        RESULTS_DIR
        / "training_result.json"
    )

    for output_file in [
        experiment_file,
        result_file,
    ]:

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

    print(
        "\nTraining summary saved."
    )

    print(
        "\nTRAINING PIPELINE TEST PASSED."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()