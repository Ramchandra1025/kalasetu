import json

from .config import (
    DATASET_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    MODEL_NAME,
    RESEARCH_EXPERIMENT_DIR,
    RESEARCH_VALID,
)
from .dataset import load_dataset
from .features import encode_features
from .model import InventoryIntelligenceModel


def create_inventory_label(row):

    current_stock = row["current_stock"]
    minimum_stock = row["minimum_stock"]
    maximum_stock = row["maximum_stock"]
    daily_sales = max(
        row["recent_daily_sales"],
        0.1,
    )

    coverage_days = (
        current_stock / daily_sales
    )

    if coverage_days <= 3:
        return "CRITICAL"

    if coverage_days <= 7:
        return "LOW_STOCK"

    if coverage_days >= 60:
        return "OVERSTOCK"

    return "HEALTHY"


def main():

    print()
    print("=" * 60)
    print("KALASETU - INVENTORY INTELLIGENCE AI TRAINING")
    print("=" * 60)

    df = load_dataset(DATASET_PATH)

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")

    df["stock_status"] = df.apply(
        create_inventory_label,
        axis=1,
    )

    X = encode_features(df)

    y = df["stock_status"]

    print(
        f"Feature matrix: {X.shape}"
    )

    print()
    print("Class distribution:")

    print(
        y.value_counts().to_string()
    )

    model = InventoryIntelligenceModel()

    model.fit(
        X,
        y,
    )

    model.save(MODEL_PATH)

    RESEARCH_EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": "XGBoost Classifier",
        "dataset_path": str(DATASET_PATH),
        "dataset_rows": len(df),
        "feature_count": X.shape[1],
        "classes": sorted(
            y.unique().tolist()
        ),
        "dataset_type": "synthetic",
        "research_valid": RESEARCH_VALID,
    }

    metadata_path = (
        RESEARCH_EXPERIMENT_DIR
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
        "Inventory model saved to:"
    )
    print(MODEL_PATH)

    print()
    print(
        "Research metadata saved to:"
    )
    print(metadata_path)

    print()
    print(
        f"Research valid: {RESEARCH_VALID}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()