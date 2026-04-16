def generate_explanation(row):
    reasons = []

    prob = float(row.get("fake_probability", 0))
    verified_purchase = int(row.get("verified_purchase", 1))
    is_extreme_rating = int(row.get("is_extreme_rating", 0))
    is_short_review = int(row.get("is_short_review", 0))
    suspicious_cluster_flag = int(row.get("suspicious_cluster_flag", 0))
    exclamation_count = int(row.get("exclamation_count", 0))
    uppercase_ratio = float(row.get("uppercase_ratio", 0))
    spam_word_count = int(row.get("spam_word_count", 0))
    repeated_word_ratio = float(row.get("repeated_word_ratio", 0))
    rating_deviation = float(row.get("rating_deviation", 0))
    reviewer_daily_review_count = int(row.get("reviewer_daily_review_count", 1))
    product_daily_review_count = int(row.get("product_daily_review_count", 1))
    invalid_input = bool(row.get("invalid_input", False))

    # Highest priority: invalid or meaningless input
    if invalid_input:
        return "Suspicious because the submitted text does not appear to be a meaningful review."

    # Strong suspicious signals
    if verified_purchase == 0:
        reasons.append("review is not from a verified purchase")

    if is_short_review == 1 and prob >= 0.4:
        reasons.append("review is very short")

    if is_extreme_rating == 1 and verified_purchase == 0:
        reasons.append("uses an extreme rating without verified purchase")

    if rating_deviation > 2:
        reasons.append("rating differs strongly from the product average")

    if reviewer_daily_review_count > 3:
        reasons.append("reviewer posted many reviews on the same day")

    if product_daily_review_count > 10:
        reasons.append("product received unusually high review activity in one day")

    if suspicious_cluster_flag == 1:
        reasons.append("belongs to a suspicious reviewer cluster")

    if exclamation_count >= 3:
        reasons.append("uses excessive exclamation marks")

    if uppercase_ratio > 0.3:
        reasons.append("uses a high amount of uppercase text")

    if spam_word_count >= 1:
        reasons.append("contains promotional or spam-like phrases")

    if repeated_word_ratio > 0.4:
        reasons.append("repeats words unusually often")

    # Add model-level explanation only when probability is clearly elevated
    if prob >= 0.6:
        reasons.append("model detected patterns similar to fake reviews")

    # Final wording by risk band
    if prob >= 0.7:
        if reasons:
            return "Suspicious because " + ", ".join(reasons) + "."
        return "Suspicious because the model assigned a high fake probability."

    if prob >= 0.5:
        if reasons:
            return "Potentially suspicious because " + ", ".join(reasons) + "."
        return "Potentially suspicious because the model detected mixed signals."

    if prob >= 0.35:
        if reasons:
            return "Mostly authentic, but some mild suspicious signals were detected: " + ", ".join(reasons) + "."
        return "Mostly authentic, but the model detected some mild suspicious patterns."

    if reasons:
        return "Likely authentic overall, although minor signals were noted: " + ", ".join(reasons) + "."

    return "Likely authentic with no major suspicious signals detected."