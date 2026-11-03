import json
from pathlib import Path

from .dataset import (
    load_dataset,
    prepare_dataset,
    validate_dataset,
)

from .features import (
    build_feature_matrix,
)

from .model import (
    AnomalyDetector,
)

from .config import (
    MODEL_PATH,
    RESEARCH_EXPERIMENT_DIR,
    MODEL_NAME,
    MODEL_VERSION,
    DATASET_TYPE,
    RESEARCH_VALID,
)


def train():

    print("=" * 60)
    print("KALASETU - ANOMALY DETECTION TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    validation = validate_dataset(df)

    print(
        f"Dataset rows: {validation['rows']}"
    )

    print(
        f"Dataset columns: {validation['columns']}"
    )

    if not validation["valid"]:

        raise ValueError(
            f"Invalid dataset: {validation}"
        )

    df = prepare_dataset(df)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    X = build_feature_matrix(df)

    print(
        f"Feature matrix: {X.shape}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    detector = AnomalyDetector()

    detector.fit(X)

    # --------------------------------------------------------
    # Training predictions
    # --------------------------------------------------------

    predictions = detector.predict(X)

    anomaly_scores = (
        detector.anomaly_score(X)
    )

    anomaly_count = int(
        (predictions == -1).sum()
    )

    normal_count = int(
        (predictions == 1).sum()
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    detector.save(MODEL_PATH)

    # --------------------------------------------------------
    # Research metadata
    # --------------------------------------------------------

    metadata = {

        "model_name": MODEL_NAME,

        "model_version": MODEL_VERSION,

        "algorithm": "Isolation Forest",

        "dataset_type": DATASET_TYPE,

        "research_valid": RESEARCH_VALID,

        "rows": len(df),

        "feature_count": X.shape[1],

        "anomaly_count": anomaly_count,

        "normal_count": normal_count,

        "average_risk_score": float(
            anomaly_scores.mean()
        ),

        "model_path": str(MODEL_PATH),
    }

    metadata_path = (
        Path(RESEARCH_EXPERIMENT_DIR)
        / "training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print(
        f"Normal samples: {normal_count}"
    )

    print(
        f"Anomalous samples: {anomaly_count}"
    )

    print(
        f"Average risk score: "
        f"{anomaly_scores.mean():.2f}"
    )

    print()
    print(
        f"Model saved to:\n{MODEL_PATH}"
    )

    print()
    print(
        "Research valid:",
        RESEARCH_VALID,
    )

    print("=" * 60)

    return detector


if __name__ == "__main__":
    train()