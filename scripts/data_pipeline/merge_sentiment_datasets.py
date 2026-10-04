# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
merge_sentiment_datasets.py
----------------------------
Merges data/IMDB Dataset.csv and data/tweets.csv into a single
data/merged_sentiment.csv with consistent labels.

Classification rule
-------------------
  IMDB:   "positive"  → 1  |  "negative" → 0
  Tweets: "4"         → 1  |  "0"        → 0

Output columns: text, label (0 = negative, 1 = positive)

HOW TO RUN
----------
    python merge_sentiment_datasets.py

OUTPUT
------
    data/merged_sentiment.csv  (~1,650,000 rows)
"""

import csv
import os

IMDB_FILE   = os.path.join(os.path.dirname(__file__), "data", "IMDB Dataset.csv")
TWEETS_FILE = os.path.join(os.path.dirname(__file__), "data", "tweets.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "merged_sentiment.csv")

IMDB_LABEL_MAP   = {"positive": 1, "negative": 0}
TWEETS_LABEL_MAP = {"4": 1, "0": 0}


def main():
    total_pos = 0
    total_neg = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["text", "label"])

        # --- IMDB ---
        print(f"Reading IMDB data from: {IMDB_FILE}")
        imdb_pos = imdb_neg = 0
        with open(IMDB_FILE, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                sentiment = row.get("sentiment", "").strip().lower()
                text      = row.get("review", "").strip()
                label     = IMDB_LABEL_MAP.get(sentiment)
                if label is None or not text:
                    continue
                writer.writerow([text, label])
                if label == 1:
                    imdb_pos += 1
                else:
                    imdb_neg += 1
        print(f"  IMDB rows written: {imdb_pos + imdb_neg:,}  "
              f"({imdb_pos:,} positive / {imdb_neg:,} negative)")
        total_pos += imdb_pos
        total_neg += imdb_neg

        # --- Tweets ---
        print(f"Reading tweets data from: {TWEETS_FILE}")
        tweet_pos = tweet_neg = 0
        with open(TWEETS_FILE, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                raw_label = row[0].strip()
                text      = row[5].strip()
                label     = TWEETS_LABEL_MAP.get(raw_label)
                if label is None or not text:
                    continue
                writer.writerow([text, label])
                if label == 1:
                    tweet_pos += 1
                else:
                    tweet_neg += 1
        print(f"  Tweets rows written: {tweet_pos + tweet_neg:,}  "
              f"({tweet_pos:,} positive / {tweet_neg:,} negative)")
        total_pos += tweet_pos
        total_neg += tweet_neg

    total = total_pos + total_neg
    print()
    print(f"Merged CSV saved to: {OUTPUT_FILE}")
    print(f"Total rows : {total:,}")
    print(f"  Positive : {total_pos:,}")
    print(f"  Negative : {total_neg:,}")


if __name__ == "__main__":
    main()
