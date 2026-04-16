def generate_explanation(row):
    reasons = []

    # Feature-based reasons
    if row.get("is_short_review", 0) == 1:
        reasons.append("review is very short")

    if row.get("verified_purchase", 1) == 0:
        reasons.append("review is not from a verified purchase")

    if row.get("is_extreme_rating", 0) == 1:
        reasons.append("uses an extreme rating")

    if row.get("rating_deviation", 0) > 2:
        reasons.append("rating differs strongly from the product average")

    if row.get("reviewer_daily_review_count", 1) > 3:
        reasons.append("reviewer posted many reviews on the same day")

    if row.get("product_daily_review_count", 1) > 10:
        reasons.append("product received unusually high review activity in one day")

    if row.get("suspicious_cluster_flag", 0) == 1:
        reasons.append("belongs to a suspicious reviewer cluster")

    prob = row.get("fake_probability", 0)

    if prob >= 0.6:
        if reasons:
            return "Suspicious because " + ", ".join(reasons) + "."
        return "Suspicious because the model assigned a high fake probability."

    if prob >= 0.4:
        if reasons:
            return "Mostly authentic, but some mild suspicious signals were detected: " + ", ".join(reasons) + "."
        return "Mostly authentic, but the model detected some mild suspicious patterns."

    if reasons:
        return "Likely authentic overall, although minor signals were noted: " + ", ".join(reasons) + "."

    return "Likely authentic with no major suspicious signals detected."