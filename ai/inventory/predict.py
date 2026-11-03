import argparse
import json

import pandas as pd

from .config import (
    MODEL_PATH,
    MODEL_VERSION,
    MODEL_NAME,
    RESEARCH_VALID,
)

from .features import encode_features
from .model import InventoryIntelligenceModel


def calculate_inventory_intelligence(data):

    current_stock = float(
        data["current_stock"]
    )

    daily_sales = max(
        float(data["recent_daily_sales"]),
        0.1,
    )

    lead_time = float(
        data["lead_time_days"]
    )

    stock_coverage_days = (
        current_stock / daily_sales
    )

    safety_stock = (
        daily_sales * 3
    )

    reorder_point = (
        daily_sales * lead_time
        + safety_stock
    )

    recommended_stock = max(
        float(data["maximum_stock"]),
        reorder_point,
    )

    reorder_quantity = max(
        0,
        recommended_stock - current_stock,
    )

    reasons = []

    if stock_coverage_days <= 3:
        reasons.append(
            "inventory may run out within three days"
        )

    elif stock_coverage_days <= 7:
        reasons.append(
            "low stock coverage"
        )

    if daily_sales > float(
        data["average_daily_sales"]
    ):
        reasons.append(
            "recent sales are above average"
        )

    if float(
        data["demand_growth"]
    ) > 0:
        reasons.append(
            "demand is growing"
        )

    if current_stock < reorder_point:
        reasons.append(
            "current stock is below reorder point"
        )

    if stock_coverage_days >= 60:
        reasons.append(
            "inventory coverage is unusually high"
        )

    if not reasons:
        reasons.append(
            "inventory level appears balanced"
        )

    return {
        "stock_coverage_days": round(
            stock_coverage_days,
            2,
        ),
        "safety_stock": round(
            safety_stock,
            2,
        ),
        "reorder_point": round(
            reorder_point,
            2,
        ),
        "recommended_stock_level": round(
            recommended_stock,
            2,
        ),
        "recommended_reorder_quantity": round(
            reorder_quantity,
            2,
        ),
        "reasons": reasons,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "KalaSetu Inventory Intelligence prediction"
        )
    )

    parser.add_argument(
        "--category",
        default="wood_craft",
    )

    parser.add_argument(
        "--material",
        default="wood",
    )

    parser.add_argument(
        "--region",
        default="Karnataka",
    )

    parser.add_argument(
        "--stock",
        type=float,
        default=12,
    )

    parser.add_argument(
        "--minimum-stock",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--maximum-stock",
        type=float,
        default=50,
    )

    parser.add_argument(
        "--average-sales",
        type=float,
        default=6,
    )

    parser.add_argument(
        "--recent-sales",
        type=float,
        default=8,
    )

    parser.add_argument(
        "--historical-demand",
        type=float,
        default=70,
    )

    parser.add_argument(
        "--demand-growth",
        type=float,
        default=8,
    )

    parser.add_argument(
        "--lead-time",
        type=float,
        default=7,
    )

    parser.add_argument(
        "--price",
        type=float,
        default=1200,
    )

    parser.add_argument(
        "--days-last-order",
        type=float,
        default=5,
    )

    parser.add_argument(
        "--stockouts",
        type=float,
        default=1,
    )

    args = parser.parse_args()

    row = pd.DataFrame(
        [
            {
                "inventory_id": "prediction",
                "product_category": args.category,
                "material": args.material,
                "region": args.region,
                "current_stock": args.stock,
                "minimum_stock": args.minimum_stock,
                "maximum_stock": args.maximum_stock,
                "average_daily_sales": args.average_sales,
                "recent_daily_sales": args.recent_sales,
                "historical_demand": args.historical_demand,
                "demand_growth": args.demand_growth,
                "lead_time_days": args.lead_time,
                "unit_price": args.price,
                "days_since_last_order": args.days_last_order,
                "stockout_count": args.stockouts,
            }
        ]
    )

    model = InventoryIntelligenceModel.load(
        MODEL_PATH
    )

    X = encode_features(row)

    predicted_status = model.predict(X).iloc[0]

    probabilities = model.predict_proba(X)[0]

    class_names = model.class_names

    class_confidence = dict(
        zip(
            class_names,
            [
                round(
                    float(probability) * 100,
                    2,
                )
                for probability in probabilities
            ],
        )
    )

    intelligence = calculate_inventory_intelligence(
        row.iloc[0]
    )

    if predicted_status == "CRITICAL":
        action = "RESTOCK IMMEDIATELY"

    elif predicted_status == "LOW_STOCK":
        action = "REORDER SOON"

    elif predicted_status == "OVERSTOCK":
        action = "REDUCE NEW PRODUCTION"

    else:
        action = "MAINTAIN CURRENT INVENTORY"

    result = {
        "inventory_id": "prediction",
        "product_category": args.category,
        "material": args.material,
        "region": args.region,
        "current_stock": args.stock,
        "stock_status": predicted_status,
        "confidence": class_confidence,
        "stock_coverage_days": intelligence[
            "stock_coverage_days"
        ],
        "safety_stock": intelligence[
            "safety_stock"
        ],
        "reorder_point": intelligence[
            "reorder_point"
        ],
        "recommended_stock_level": intelligence[
            "recommended_stock_level"
        ],
        "recommended_reorder_quantity": intelligence[
            "recommended_reorder_quantity"
        ],
        "recommended_action": action,
        "reasons": intelligence[
            "reasons"
        ],
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "research_valid": RESEARCH_VALID,
        "note": (
            "Prototype inventory intelligence using "
            "synthetic data. Recommendations are "
            "decision-support estimates and should "
            "not be treated as guaranteed demand."
        ),
    }

    print()

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()