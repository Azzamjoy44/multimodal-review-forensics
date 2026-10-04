import json
import os

# Path to the reviews file, relative to this script's location
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_reviews.json")


def load_reviews():
    """
    Load all reviews from the JSON file and return them as a list of dicts.
    Returns an empty list if the file is missing or the JSON is invalid.
    """
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            reviews = json.load(f)
        return reviews
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DATA_FILE}'.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON — {e}")
        return []


def get_reviews_by_product(product_name):
    """
    Return only the reviews that match the given product name.
    Matching is case-insensitive so 'cyberpunk 2077' == 'Cyberpunk 2077'.
    """
    all_reviews = load_reviews()

    # Normalise the search term once before the loop
    search = product_name.strip().lower()

    matching = [
        review for review in all_reviews
        if review.get("product_name", "").lower() == search
    ]

    return matching


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see it working:
#   python load_reviews.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== All reviews ===")
    all_reviews = load_reviews()
    print(f"Total reviews loaded: {len(all_reviews)}\n")

    print("=== Reviews for 'Cyberpunk 2077' ===")
    cp_reviews = get_reviews_by_product("Cyberpunk 2077")
    for r in cp_reviews:
        print(f"  [{r['review_id']}] rating={r['rating']}  score={r['suspicion_score']}  \"{r['review_text'][:60]}...\"")

    print("\n=== Case-insensitive check: 'cyberpunk 2077' ===")
    cp_lower = get_reviews_by_product("cyberpunk 2077")
    print(f"  Found {len(cp_lower)} reviews (same result as above)")

    print("\n=== Unknown product ===")
    unknown = get_reviews_by_product("Nonexistent Game")
    print(f"  Found {len(unknown)} reviews")
