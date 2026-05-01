import numpy as np
import json

# ── PAKISTANI BRAND SIZE CHARTS ───────────────────────
# Measurements in centimeters (chest, waist, length)
SIZE_CHARTS = {
    "Khaadi": {
        "XS": {"chest": (76, 81),  "waist": (61, 66),  "length": (96, 99)},
        "S":  {"chest": (81, 86),  "waist": (66, 71),  "length": (99, 102)},
        "M":  {"chest": (86, 91),  "waist": (71, 76),  "length": (102, 105)},
        "L":  {"chest": (91, 97),  "waist": (76, 81),  "length": (105, 108)},
        "XL": {"chest": (97, 102), "waist": (81, 86),  "length": (108, 111)},
        "XXL":{"chest": (102,107), "waist": (86, 91),  "length": (111, 114)},
    },
    "Gul Ahmed": {
        "XS": {"chest": (78, 82),  "waist": (62, 66),  "length": (95, 98)},
        "S":  {"chest": (82, 87),  "waist": (66, 70),  "length": (98, 101)},
        "M":  {"chest": (87, 92),  "waist": (70, 75),  "length": (101, 104)},
        "L":  {"chest": (92, 97),  "waist": (75, 80),  "length": (104, 107)},
        "XL": {"chest": (97, 102), "waist": (80, 85),  "length": (107, 110)},
        "XXL":{"chest": (102,107), "waist": (85, 90),  "length": (110, 113)},
    },
    "Alkaram": {
        "XS": {"chest": (77, 81),  "waist": (61, 65),  "length": (94, 97)},
        "S":  {"chest": (81, 86),  "waist": (65, 70),  "length": (97, 100)},
        "M":  {"chest": (86, 91),  "waist": (70, 75),  "length": (100, 103)},
        "L":  {"chest": (91, 96),  "waist": (75, 80),  "length": (103, 106)},
        "XL": {"chest": (96, 101), "waist": (80, 85),  "length": (106, 109)},
        "XXL":{"chest": (101,106), "waist": (85, 90),  "length": (109, 112)},
    }
}

def calculate_fit_score(chest, waist, length, brand, size):
    """
    Calculate how well user measurements fit a specific size
    Returns score 0-100
    """
    chart = SIZE_CHARTS[brand][size]

    scores = []
    measurements = {
        "chest": chest,
        "waist": waist,
        "length": length
    }

    for measurement, value in measurements.items():
        min_val, max_val = chart[measurement]
        mid_val = (min_val + max_val) / 2
        range_val = (max_val - min_val) / 2

        if min_val <= value <= max_val:
            # Inside range — calculate how centered it is
            distance = abs(value - mid_val)
            score = 100 - (distance / range_val) * 30
        else:
            # Outside range — penalize based on distance
            if value < min_val:
                distance = min_val - value
            else:
                distance = value - max_val
            score = max(0, 100 - distance * 10)

        scores.append(score)

    return round(np.mean(scores))


def recommend_size(chest, waist, length, brand):
    """
    Recommend best size for given measurements and brand
    Returns size, confidence score, and breakdown
    """
    if brand not in SIZE_CHARTS:
        return None, 0, {}

    chart = SIZE_CHARTS[brand]
    best_size = None
    best_score = 0
    all_scores = {}
    breakdown = {}

    for size in chart:
        score = calculate_fit_score(chest, waist, length, brand, size)
        all_scores[size] = score

        if score > best_score:
            best_score = score
            best_size = size

    # Build breakdown for best size
    size_chart = chart[best_size]
    breakdown = {
        "chest": {
            "your_measurement": chest,
            "size_range": size_chart["chest"],
            "fits": size_chart["chest"][0] <= chest <= size_chart["chest"][1]
        },
        "waist": {
            "your_measurement": waist,
            "size_range": size_chart["waist"],
            "fits": size_chart["waist"][0] <= waist <= size_chart["waist"][1]
        },
        "length": {
            "your_measurement": length,
            "size_range": size_chart["length"],
            "fits": size_chart["length"][0] <= length <= size_chart["length"][1]
        }
    }

    return best_size, best_score, breakdown, all_scores


def generate_explanation(chest, waist, length, brand):
    """
    Generate plain English explanation of fit recommendation
    """
    size, score, breakdown, all_scores = recommend_size(
        chest, waist, length, brand
    )

    if not size:
        return "Brand not found in our size charts."

    explanation = f"Based on your measurements, we recommend size {size} "
    explanation += f"from {brand} with {score}% confidence.\n\n"

    for measurement, details in breakdown.items():
        your_val = details["your_measurement"]
        min_val, max_val = details["size_range"]
        fits = details["fits"]

        if fits:
            explanation += f"✅ {measurement.capitalize()} ({your_val}cm): "
            explanation += f"fits well within {size} range "
            explanation += f"({min_val}-{max_val}cm)\n"
        else:
            if your_val < min_val:
                explanation += f"⚠️ {measurement.capitalize()} ({your_val}cm): "
                explanation += f"slightly small for {size} "
                explanation += f"({min_val}-{max_val}cm)\n"
            else:
                explanation += f"⚠️ {measurement.capitalize()} ({your_val}cm): "
                explanation += f"slightly large for {size} "
                explanation += f"({min_val}-{max_val}cm)\n"

    return explanation


# ── TEST ───────────────────────────────────────────────
if __name__ == "__main__":
    # Test measurements
    chest = 88
    waist = 72
    length = 103

    print("=== FIT SCORE ENGINE TEST ===\n")

    for brand in SIZE_CHARTS.keys():
        print(f"Brand: {brand}")
        size, score, breakdown, all_scores = recommend_size(
            chest, waist, length, brand
        )
        print(f"Recommended Size: {size}")
        print(f"Confidence: {score}%")
        print(f"All scores: {all_scores}")

        explanation = generate_explanation(chest, waist, length, brand)
        print(f"\nExplanation:\n{explanation}")
        print("-" * 40)