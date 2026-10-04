import requests
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Steam API URLs
# ---------------------------------------------------------------------------
# Returns review data for a given app ID
REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"

# Returns store details (used only to look up the product name)
DETAILS_URL = "https://store.steampowered.com/api/appdetails"


# ---------------------------------------------------------------------------
# Step 1 (optional): look up the product name from Steam's store details API.
# If the request fails we just return None and the caller can supply a name.
# ---------------------------------------------------------------------------
def get_product_name(app_id):
    """
    Fetch the display name of a Steam game by its numeric app ID.
    Returns the name string, or None if the lookup fails.
    """
    try:
        response = requests.get(
            DETAILS_URL,
            params={"appids": app_id, "filters": "basic"},
            timeout=10,
        )
        response.raise_for_status()          # raise an error for 4xx / 5xx
        data = response.json()

        # The API returns { "<app_id>": { "success": true, "data": { ... } } }
        app_data = data.get(str(app_id), {})
        if app_data.get("success") and "data" in app_data:
            return app_data["data"].get("name")

    except requests.RequestException as e:
        print(f"Warning: Could not fetch product name for app {app_id}: {e}")

    return None


# ---------------------------------------------------------------------------
# Step 2 (private helper): convert one raw Steam review dict into our schema.
# ---------------------------------------------------------------------------
def _normalize_review(raw_review, app_id, product_name):
    """
    Map a single raw Steam review object to the project's review schema.

    Steam uses thumbs up / thumbs down instead of 1–5 stars, so we map:
      voted_up = True  →  rating 5
      voted_up = False →  rating 1
    """
    # Convert Unix timestamp → "YYYY-MM-DD" string
    timestamp = raw_review.get("timestamp_created", 0)
    review_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    return {
        "product_id":   f"steam_app_{app_id}",
        "product_name": product_name,
        "review_id":    f"steam_{raw_review.get('recommendationid', 'unknown')}",
        "review_text":  raw_review.get("review", "").strip(),
        "rating":       5 if raw_review.get("voted_up") else 1,
        "review_date":  review_date,
        "reviewer_id":  raw_review.get("author", {}).get("steamid", "unknown"),
        "source":       "steam",
    }


# ---------------------------------------------------------------------------
# Step 3 (main function): fetch and normalise reviews for a Steam app.
# ---------------------------------------------------------------------------
def fetch_steam_reviews(app_id, product_name=None, count=20):
    """
    Fetch up to `count` English-language reviews for the given Steam app ID.

    Parameters
    ----------
    app_id       : int or str  — Steam numeric app ID (e.g. 1091500)
    product_name : str or None — supply the name directly, or leave None to
                                 let the function look it up automatically
    count        : int         — how many reviews to fetch (max 100 per request)

    Returns
    -------
    list of dicts, each matching the project's review schema.
    Returns an empty list on failure.
    """
    # Auto-fetch the product name if not supplied
    if product_name is None:
        product_name = get_product_name(app_id) or f"steam_app_{app_id}"

    # Steam allows up to 100 reviews per request
    count = min(count, 100)

    try:
        response = requests.get(
            REVIEWS_URL.format(app_id=app_id),
            params={
                "json":         1,
                "language":     "english",
                "filter":       "recent",   # most recent reviews first
                "num_per_page": count,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        print(f"Error: Failed to fetch reviews from Steam: {e}")
        return []

    # Check that Steam returned a successful response
    if data.get("success") != 1:
        print("Error: Steam API returned a non-success response.")
        return []

    raw_reviews = data.get("reviews", [])

    if not raw_reviews:
        print(f"No reviews returned by Steam for app {app_id}.")
        return []

    # Normalise every raw review into our schema, skipping any that are empty
    normalized = []
    for raw in raw_reviews:
        review = _normalize_review(raw, app_id, product_name)

        # Skip reviews with no actual text
        if not review["review_text"]:
            continue

        normalized.append(review)

    return normalized


# ---------------------------------------------------------------------------
# Quick demo — run this file directly:
#   python fetch_steam_reviews.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Cyberpunk 2077 — a well-known app with a large number of English reviews
    APP_ID = 1091500

    print(f"Fetching reviews for Steam app {APP_ID}...\n")
    reviews = fetch_steam_reviews(APP_ID, count=5)

    if not reviews:
        print("No reviews fetched. Check your internet connection.")
    else:
        print(f"Fetched {len(reviews)} review(s).\n")
        print("=" * 60)
        for r in reviews:
            print(f"review_id   : {r['review_id']}")
            print(f"product_name: {r['product_name']}")
            print(f"rating      : {r['rating']} / 5")
            print(f"review_date : {r['review_date']}")
            print(f"reviewer_id : {r['reviewer_id']}")
            print(f"review_text : {r['review_text'][:120]}...")
            print("-" * 60)
