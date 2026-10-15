"""Severity = 50% damage type + 35% damage size + 15% detection confidence (slide 17).

LOW < 0.42 <= MEDIUM < 0.72 <= HIGH. If your earlier code used different type weights,
change TYPE_WEIGHT so the numbers in the app match the ones in your report.
"""

TYPE_WEIGHT = {"pothole": 1.00, "alligator_crack": 0.75, "transverse_crack": 0.45, "longitudinal_crack": 0.40}
W_TYPE, W_SIZE, W_CONF = 0.50, 0.35, 0.15
SIZE_FULL = 0.15        # a box covering 15% of the photo (or more) gets the maximum size score
LOW_MAX, MEDIUM_MAX = 0.42, 0.72

LABELS = {
    "pothole": "Pothole",
    "alligator_crack": "Alligator crack",
    "transverse_crack": "Transverse crack",
    "longitudinal_crack": "Longitudinal crack",
}


def canonical_class(name):
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if n == "d00" or "longitud" in n:
        return "longitudinal_crack"
    if n == "d10" or "transvers" in n:
        return "transverse_crack"
    if n == "d20" or "alligator" in n:
        return "alligator_crack"
    if n == "d40" or "pothole" in n:
        return "pothole"
    return n


def size_score(area_ratio):
    return min(1.0, max(0.0, area_ratio) / SIZE_FULL)


def severity(cls, area_ratio, conf):
    score = W_TYPE * TYPE_WEIGHT.get(cls, 0.4) + W_SIZE * size_score(area_ratio) + W_CONF * float(conf)
    score = round(min(1.0, score), 3)
    label = "LOW" if score < LOW_MAX else ("MEDIUM" if score < MEDIUM_MAX else "HIGH")
    return score, label


SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# ----------------------------------------------------------------------------- pothole box (seeing + feeling)
# The box also measures how hard the vehicle was jolted. The jolt limit already grows with speed on the box,
# so a jolt exactly at the limit scores 0.5 and a jolt twice the limit (or more) scores 1.
JOLT_WEIGHT = 0.30      # share of the final score that comes from the jolt when the camera also saw the pothole


def label_for(score):
    return "LOW" if score < LOW_MAX else ("MEDIUM" if score < MEDIUM_MAX else "HIGH")


def jolt_score(jolt_g, limit_g):
    if not jolt_g or not limit_g or limit_g <= 0:
        return None
    return round(min(1.0, max(0.0, float(jolt_g) / (2.0 * float(limit_g)))), 3)


def fuse(photo_score, jscore):
    """Camera only -> photo score. Jolt only -> jolt score. Both -> 70% photo + 30% jolt."""
    if jscore is None:
        score = photo_score
    elif photo_score is None:
        score = jscore
    else:
        score = (1 - JOLT_WEIGHT) * photo_score + JOLT_WEIGHT * jscore
    score = round(min(1.0, score), 3)
    return score, label_for(score)
