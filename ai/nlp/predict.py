import argparse
import json

from sklearn.metrics.pairwise import cosine_similarity

from .config import RESEARCH_VALID

from .features import (
    clean_text,
    detect_category,
    description_quality,
    extract_keywords,
)

from .model import NLPModel


def predict(
    text: str,
):

    cleaned = clean_text(
        text
    )

    if not cleaned:
        raise ValueError(
            "Product description cannot be empty."
        )

    model = NLPModel.load()

    matrix = model.transform(
        [cleaned]
    )

    category, category_confidence = (
        detect_category(
            cleaned
        )
    )

    quality = description_quality(
        cleaned
    )

    keywords = extract_keywords(
        cleaned,
        model.vectorizer,
        matrix[0],
    )

    # --------------------------------------------------------
    # Similarity against training vocabulary
    # --------------------------------------------------------

    training_matrix = (
        model.vectorizer.transform(
            [cleaned]
        )
    )

    self_similarity = cosine_similarity(
        matrix,
        training_matrix,
    )[0][0]

    result = {
        "text": text,
        "cleaned_text": cleaned,
        "detected_category": category,
        "category_confidence": category_confidence,
        "keywords": keywords,
        "description_quality": quality,
        "semantic_representation_size": (
            matrix.shape[1]
        ),
        "self_similarity": round(
            float(self_similarity),
            4,
        ),
        "model": "TF-IDF",
        "model_version": "1.0.0",
        "research_valid": RESEARCH_VALID,
        "note": (
            "Prototype NLP intelligence using "
            "synthetic data. Semantic results are "
            "not research-valid until evaluated "
            "on a representative real dataset."
        ),
    }

    return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "KalaSetu NLP / Semantic Intelligence"
        )
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Product description",
    )

    args = parser.parse_args()

    result = predict(
        args.text
    )

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()