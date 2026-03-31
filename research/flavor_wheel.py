"""
SCA Coffee Taster's Flavor Wheel — Hierarchical Taxonomy
=========================================================
Encodes the official SCA flavor wheel as a structured taxonomy
for tasting note classification and flavor profile prediction.
"""

# Level 1 → Level 2 → Level 3 descriptors
FLAVOR_WHEEL = {
    "Floral": {
        "Floral": ["jasmine", "rose", "chamomile", "lavender"],
        "Black Tea": ["earl_grey", "bergamot"],
    },
    "Fruity": {
        "Berry": ["blackberry", "raspberry", "blueberry", "strawberry"],
        "Dried Fruit": ["raisin", "prune", "fig", "date"],
        "Citrus": ["lemon", "lime", "orange", "grapefruit", "tangerine"],
        "Stone Fruit": ["peach", "nectarine", "apricot", "cherry", "plum"],
        "Tropical": ["pineapple", "mango", "passion_fruit", "guava", "lychee", "papaya"],
        "Other Fruit": ["apple", "pear", "grape", "pomegranate"],
    },
    "Sour/Fermented": {
        "Sour": ["citric", "malic", "tartaric", "phosphoric", "acetic"],
        "Fermented": ["winey", "boozy", "overripe", "vinegar"],
    },
    "Green/Vegetal": {
        "Olive Oil": ["olive_oil"],
        "Raw": ["green_pepper", "pea", "cucumber"],
        "Herb-like": ["thyme", "basil", "mint", "sage"],
        "Beany": ["raw_bean"],
    },
    "Roasted": {
        "Cereal": ["malt", "grain", "toast"],
        "Burnt": ["smoky", "ashy", "carbon", "burnt"],
        "Tobacco": ["pipe_tobacco", "cigar"],
    },
    "Spices": {
        "Brown Spice": ["cinnamon", "nutmeg", "clove", "cardamom", "allspice"],
        "Pepper": ["black_pepper", "pink_pepper"],
        "Pungent": ["ginger", "anise"],
    },
    "Nutty/Cocoa": {
        "Nutty": ["almond", "hazelnut", "peanut", "walnut", "cashew", "macadamia"],
        "Cocoa": ["dark_chocolate", "milk_chocolate", "cocoa_powder", "cacao_nib"],
    },
    "Sweet": {
        "Sugar": ["brown_sugar", "raw_sugar", "molasses", "maple_syrup", "caramel", "toffee"],
        "Vanilla": ["vanilla", "vanillin"],
        "Honey": ["honey", "honeycomb"],
        "Overall Sweet": ["sweet", "sugarcane"],
    },
    "Other": {
        "Chemical": ["rubber", "medicinal", "salty"],
        "Papery/Musty": ["stale", "cardboard", "woody", "earthy", "musty"],
    },
}

# Flatten to get all descriptors
ALL_DESCRIPTORS = []
DESCRIPTOR_TO_L1 = {}
DESCRIPTOR_TO_L2 = {}
for l1, l2_dict in FLAVOR_WHEEL.items():
    for l2, descriptors in l2_dict.items():
        for d in descriptors:
            ALL_DESCRIPTORS.append(d)
            DESCRIPTOR_TO_L1[d] = l1
            DESCRIPTOR_TO_L2[d] = l2

NUM_DESCRIPTORS = len(ALL_DESCRIPTORS)

# Acidity character types (orthogonal to flavor wheel)
ACIDITY_TYPES = ["bright", "crisp", "juicy", "sparkling", "winey", "tart", "soft", "flat"]

# Body character types
BODY_TYPES = ["tea_like", "silky", "syrupy", "creamy", "full", "heavy", "thin", "watery"]

# Finish/aftertaste types
FINISH_TYPES = ["clean", "lingering", "dry", "sweet", "complex"]


def descriptors_to_vector(descriptors: list[str]) -> list[float]:
    """Convert a list of descriptor strings to a binary vector."""
    vec = [0.0] * NUM_DESCRIPTORS
    for d in descriptors:
        d_lower = d.lower().replace(" ", "_").replace("-", "_")
        if d_lower in ALL_DESCRIPTORS:
            vec[ALL_DESCRIPTORS.index(d_lower)] = 1.0
    return vec


def vector_to_descriptors(vec: list[float], threshold: float = 0.5) -> list[str]:
    """Convert a probability vector back to descriptor list."""
    return [ALL_DESCRIPTORS[i] for i, v in enumerate(vec) if v >= threshold]


def flavor_profile_summary(descriptors: list[str]) -> dict:
    """Summarize descriptors into L1 category weights."""
    counts = {}
    for d in descriptors:
        d_lower = d.lower().replace(" ", "_").replace("-", "_")
        l1 = DESCRIPTOR_TO_L1.get(d_lower)
        if l1:
            counts[l1] = counts.get(l1, 0) + 1
    total = sum(counts.values()) or 1
    return {k: round(v / total, 2) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# Common variety → typical flavor associations (domain knowledge prior)
VARIETY_FLAVOR_PRIORS = {
    "Gesha": ["jasmine", "bergamot", "peach", "honey", "tea_like"],
    "Ethiopian Heirloom": ["blueberry", "jasmine", "citric", "honey", "tea_like"],
    "SL28": ["blackberry", "grapefruit", "phosphoric", "brown_sugar", "juicy"],
    "SL34": ["plum", "brown_sugar", "malic", "full"],
    "Bourbon": ["caramel", "brown_sugar", "apple", "milk_chocolate", "creamy"],
    "Typica": ["sweet", "milk_chocolate", "almond", "clean", "soft"],
    "Caturra": ["citric", "caramel", "light", "bright"],
    "Pacamara": ["dark_chocolate", "orange", "syrupy", "complex"],
    "74158": ["jasmine", "peach", "lemon", "honey", "tea_like"],
    "Catuai": ["brown_sugar", "almond", "soft"],
    "Castillo": ["caramel", "nutty", "soft"],
    "Catimor": ["earthy", "grain", "thin"],
}

# Process → typical flavor impact (domain knowledge prior)
PROCESS_FLAVOR_IMPACT = {
    "washed": {"adds": ["clean", "bright", "citric"], "removes": ["boozy", "overripe"]},
    "natural": {"adds": ["strawberry", "blueberry", "boozy", "winey", "heavy"], "removes": ["clean"]},
    "honey_yellow": {"adds": ["honey", "sweet", "soft"], "removes": []},
    "honey_red": {"adds": ["cherry", "brown_sugar", "syrupy"], "removes": []},
    "honey_black": {"adds": ["raisin", "prune", "heavy", "winey"], "removes": ["bright"]},
    "wet_hulled": {"adds": ["earthy", "woody", "heavy", "pipe_tobacco"], "removes": ["bright", "floral"]},
}


def predict_flavor_prior(variety: str, process: str) -> list[str]:
    """Predict likely tasting notes based on variety + process domain knowledge."""
    notes = list(VARIETY_FLAVOR_PRIORS.get(variety, ["sweet", "brown_sugar"]))
    impact = PROCESS_FLAVOR_IMPACT.get(process, {})
    notes.extend(impact.get("adds", []))
    for remove in impact.get("removes", []):
        if remove in notes:
            notes.remove(remove)
    # Deduplicate while preserving order
    seen = set()
    return [n for n in notes if not (n in seen or seen.add(n))]


if __name__ == "__main__":
    print(f"SCA Flavor Wheel: {NUM_DESCRIPTORS} descriptors")
    print(f"L1 categories: {list(FLAVOR_WHEEL.keys())}")
    print(f"\nExample — Gesha washed:")
    notes = predict_flavor_prior("Gesha", "washed")
    print(f"  Notes: {notes}")
    print(f"  Profile: {flavor_profile_summary(notes)}")
    print(f"\nExample — Ethiopian Heirloom natural:")
    notes = predict_flavor_prior("Ethiopian Heirloom", "natural")
    print(f"  Notes: {notes}")
    print(f"  Profile: {flavor_profile_summary(notes)}")
