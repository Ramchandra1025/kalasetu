"""
KalaSetu - Fair Pricing AI
Price Prediction / Inference

Purpose:
    Load the trained XGBoost model and predict a recommended
    fair price for an artisan product.

This module can later be imported directly by Flask APIs.

IMPORTANT:
    The current model is trained on synthetic pipeline-testing
    data. Predictions are therefore prototype outputs only.
"""

import argparse
import json

import numpy as np
import pandas as pd

from .config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ROUND_PRICE_TO,
    MIN_PREDICTED_PRICE,
)

from .model import (
    load_model,
    load_encoder,
    predict_price,
)


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_input(product_data, encoder):
    """
    Convert a product dictionary into the exact feature format
    expected by the trained XGBoost model.
    """

    # --------------------------------------------------------
    # Validate required fields
    # --------------------------------------------------------

    required_fields = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in product_data
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields:\n"
            + "\n".join(
                f"  - {field}"
                for field in missing_fields
            )
        )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_values = []

    for feature in NUMERIC_FEATURES:

        try:
            value = float(
                product_data[feature]
            )

        except (TypeError, ValueError):

            raise ValueError(
                f"Invalid numeric value for "
                f"'{feature}'."
            )

        numeric_values.append(value)

    numeric_array = np.asarray(
        numeric_values,
        dtype=np.float32,
    ).reshape(1, -1)

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_data = pd.DataFrame(
        [
            {
                feature: str(
                    product_data[feature]
                ).strip().lower()
                for feature in CATEGORICAL_FEATURES
            }
        ]
    )

    encoded_categories = encoder.transform(
        categorical_data
    )

    encoded_array = np.asarray(
        encoded_categories,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    features = np.concatenate(
        [
            numeric_array,
            encoded_array,
        ],
        axis=1,
    )

    return features


# ============================================================
# PRICE PREDICTION
# ============================================================

def predict_fair_price(product_data):
    """
    Predict a recommended fair price.

    Parameters
    ----------
    product_data : dict
        Product information containing:

        size_cm
        production_hours
        material_cost
        labor_cost
        demand_score
        complexity_score
        craft_category
        material
        region

    Returns
    -------
    dict
        Prediction result.
    """

    # --------------------------------------------------------
    # Load model and encoder
    # --------------------------------------------------------

    model = load_model()
    encoder = load_encoder()

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    features = prepare_input(
        product_data,
        encoder,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = predict_price(
        model,
        features,
    )

    predicted_price = float(
        prediction[0]
    )

    predicted_price = max(
        predicted_price,
        MIN_PREDICTED_PRICE,
    )

    # --------------------------------------------------------
    # Round price
    # --------------------------------------------------------

    if ROUND_PRICE_TO > 0:

        predicted_price = (
            round(
                predicted_price
                / ROUND_PRICE_TO
            )
            * ROUND_PRICE_TO
        )

    return {
        "recommended_price": float(
            predicted_price
        ),

        "currency": "INR",

        "model": "XGBRegressor",

        "data_type": (
            "synthetic_pipeline_testing_data"
        ),

        "research_valid": False,

        "warning": (
            "This prediction is generated by a "
            "prototype model trained on synthetic "
            "pipeline-testing data."
        ),
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KalaSetu Fair Pricing AI "
            "Prediction"
        )
    )

    parser.add_argument(
        "--craft-category",
        required=True,
        help="Craft category",
    )

    parser.add_argument(
        "--material",
        required=True,
        help="Primary material",
    )

    parser.add_argument(
        "--region",
        required=True,
        help="Production region",
    )

    parser.add_argument(
        "--size",
        required=True,
        type=float,
        help="Product size in cm",
    )

    parser.add_argument(
        "--hours",
        required=True,
        type=float,
        help="Production hours",
    )

    parser.add_argument(
        "--material-cost",
        required=True,
        type=float,
        help="Material cost in INR",
    )

    parser.add_argument(
        "--labor-cost",
        required=True,
        type=float,
        help="Labor cost in INR",
    )

    parser.add_argument(
        "--demand",
        required=True,
        type=float,
        help="Demand score from 0 to 100",
    )

    parser.add_argument(
        "--complexity",
        required=True,
        type=float,
        help="Complexity score from 0 to 100",
    )

    args = parser.parse_args()

    product_data = {
        "craft_category": args.craft_category,
        "material": args.material,
        "region": args.region,

        "size_cm": args.size,
        "production_hours": args.hours,
        "material_cost": args.material_cost,
        "labor_cost": args.labor_cost,
        "demand_score": args.demand,
        "complexity_score": args.complexity,
    }

    print("=" * 70)
    print("KALASETU - FAIR PRICING AI")
    print("=" * 70)

    print("\nProduct Input:")

    for key, value in product_data.items():
        print(
            f"  {key:<20}: {value}"
        )

    print("\nRunning prediction...")

    result = predict_fair_price(
        product_data
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        f"\nRecommended Fair Price: "
        f"₹{result['recommended_price']:.0f}"
    )

    print(
        f"Currency               : "
        f"{result['currency']}"
    )

    print(
        f"Model                  : "
        f"{result['model']}"
    )

    print(
        f"Research Valid         : "
        f"{result['research_valid']}"
    )

    print(
        f"\nWarning:"
        f"\n{result['warning']}"
    )

    print("\nJSON Result:")

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print("\n" + "=" * 70)
    print("PREDICTION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()