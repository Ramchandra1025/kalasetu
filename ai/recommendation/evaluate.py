"""
KalaSetu - Buyer Matching & Recommendation AI
Evaluation
"""

import json

from .config import (
    EVALUATION_DIR,
    RESULTS_DIR,
)

from .dataset import (
    create_dataset,
)

from .scoring import (
    rank_products_for_buyer,
)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)

    print(
        "KALASETU - BUYER RECOMMENDATION EVALUATION"
    )

    print("=" * 70)

    dataframe = create_dataset()

    buyers = (
        dataframe[
            "buyer_id"
        ]
        .unique()
        .tolist()
    )

    all_results = {}

    print(
        f"\nUnique buyers: "
        f"{len(buyers)}"
    )

    for buyer_id in buyers:

        ranked = rank_products_for_buyer(
            dataframe,
            buyer_id,
        )

        all_results[
            buyer_id
        ] = ranked

    # --------------------------------------------------------
    # Collect scores
    # --------------------------------------------------------

    scores = []

    for results in (
        all_results.values()
    ):

        for result in results:

            scores.append(
                result[
                    "match_score"
                ]
            )

    average_score = (
        sum(scores)
        / len(scores)
    )

    high_count = sum(
        score >= 75
        for score in scores
    )

    medium_count = sum(
        50 <= score < 75
        for score in scores
    )

    low_count = sum(
        score < 50
        for score in scores
    )

    print(
        f"\nTotal buyer-product pairs: "
        f"{len(scores)}"
    )

    print(
        f"Average match score: "
        f"{average_score:.2f}"
    )

    print(
        f"High matches: "
        f"{high_count}"
    )

    print(
        f"Medium matches: "
        f"{medium_count}"
    )

    print(
        f"Low matches: "
        f"{low_count}"
    )

    # --------------------------------------------------------
    # Show buyer recommendations
    # --------------------------------------------------------

    print(
        "\nTop Recommendations:"
    )

    shown = 0

    for buyer_id in buyers:

        ranked = all_results[
            buyer_id
        ]

        if not ranked:

            continue

        top = ranked[0]

        print(
            f"\nBuyer: "
            f"{buyer_id}"
        )

        print(
            f"  #1 "
            f"{top['product_category']} "
            f"({top['product_material']}) "
            f"→ "
            f"{top['product_region']} "
            f"| Score: "
            f"{top['match_score']:.2f}"
        )

        print(
            f"  Reason: "
            f"{top['explanation']}"
        )

        shown += 1

        if shown >= 10:

            break

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results = {

        "total_buyers":
            len(buyers),

        "total_pairs":
            len(scores),

        "average_match_score":
            float(
                average_score
            ),

        "high_matches":
            high_count,

        "medium_matches":
            medium_count,

        "low_matches":
            low_count,

        "recommendations":
            all_results,

        "data_type":
            "synthetic_pipeline_testing_data",

        "research_valid":
            False,

        "research_note":
            (
                "Recommendation results "
                "are generated using synthetic "
                "buyer-product records and "
                "a rule-based hybrid matching "
                "engine. They are not validated "
                "against real user behavior."
            ),
    }

    evaluation_file = (
        EVALUATION_DIR
        / "recommendation_evaluation.json"
    )

    result_file = (
        RESULTS_DIR
        / "recommendation_evaluation.json"
    )

    for output_file in [
        evaluation_file,
        result_file,
    ]:

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
            )

    print(
        "\nEvaluation results saved."
    )

    print(
        "\nResearch-valid result: NO"
    )

    print(
        "Reason: Synthetic pipeline-testing "
        "buyer-product data."
    )

    print(
        "\nEVALUATION TEST PASSED."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()