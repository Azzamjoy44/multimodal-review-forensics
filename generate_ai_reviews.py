"""
generate_ai_reviews.py — Route B: generate diverse, topic-matched AI (machine-
generated) product reviews via the OpenAI API, to broaden the CG class of the
fake-review detector and improve cross-generator generalization.

WHY: the leave-one-generator-out test showed the detector memorises generator
fingerprints + topic (it catches an unseen generator only ~21% of the time). This
adds (a) more GPT-family generation with heavy prompt/temperature variety, and
(b) TOPIC-MATCHING — reviews are written for REAL products (Amazon Reviews 2023
metadata), the same product space our human OR reviews cover — so the model is
pushed to learn "machine-ness", not topic or one model's style. Short/casual styles
are included on purpose (also fills the short-machine-review gap).

SETUP
    pip install openai
    PowerShell:  $env:OPENAI_API_KEY = "sk-..."
RUN  (resumable — re-run to top up; appends and skips what's already done)
    python generate_ai_reviews.py
    python generate_ai_reviews.py 20000      # optional: target row count

OUTPUT: data/ai_generated_reviews.csv  (text, domain, model, rating, persona, style)
Next step: add this as a CG source in build_broadened_cgor.py, rebuild, retrain,
and re-run evaluate_cgor_generalization.py to confirm LOGO recall improved.
"""

import os
import sys
import csv
import json
import random
import time
import urllib.request

from huggingface_hub import hf_hub_url

OUT = os.path.join(os.path.dirname(__file__), "data", "ai_generated_reviews.csv")
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 12_000
SEED = 42
random.seed(SEED)

# model + temperature mix (mostly the cheap model; some variety for diversity)
MODEL_CHOICES = [
    ("gpt-4o-mini",   0.7, 0.40),
    ("gpt-4o-mini",   1.0, 0.25),
    ("gpt-4o-mini",   0.4, 0.15),
    ("gpt-3.5-turbo", 0.8, 0.12),
    ("gpt-4o",        0.7, 0.08),
]
PERSONAS = [
    "a busy parent", "a college student", "a retiree on a fixed budget",
    "a professional chef", "a skeptical first-time buyer", "a loyal repeat customer",
    "a fitness enthusiast", "a bargain hunter", "a gift-giver", "a picky eater",
    "someone who reads every ingredient label", "a small-business owner",
]
LENGTHS = [
    "in just one short sentence", "in one or two sentences",
    "in a short paragraph", "in a detailed paragraph or two",
]
STYLES = [
    "polished and well-written", "casual and conversational",
    "quick and informal, all lowercase with a couple of typos like a phone review",
    "enthusiastic and full of energy", "measured, balanced and a little critical",
    "blunt and to the point",
]
# star distribution roughly like real reviews (positive-skewed)
STARS = [5, 5, 5, 5, 4, 4, 4, 3, 2, 1, 1]

# product categories to seed from (food to match Amazon Fine Food OR, plus general)
CATS = ["Grocery_and_Gourmet_Food", "Health_and_Household", "Electronics",
        "Home_and_Kitchen", "Office_Products", "Beauty_and_Personal_Care",
        "Sports_and_Outdoors", "Pet_Supplies"]


def log(m):
    print(m, flush=True)


def collect_seeds(n_per_cat=4000):
    """Stream real product (title + short description) seeds from Amazon-2023 meta."""
    seeds = []
    for cat in CATS:
        url = hf_hub_url("McAuley-Lab/Amazon-Reviews-2023",
                         f"raw/meta_categories/meta_{cat}.jsonl", repo_type="dataset")
        got = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis"})
            with urllib.request.urlopen(req) as resp:
                for raw in resp:
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    title = (row.get("title") or "").strip()
                    if len(title) < 5:
                        continue
                    desc = ""
                    d = row.get("description") or []
                    if isinstance(d, list) and d:
                        desc = " ".join(str(x) for x in d)
                    feats = row.get("features") or []
                    if isinstance(feats, list) and feats:
                        desc = (desc + " " + " ".join(str(x) for x in feats)).strip()
                    seeds.append((title[:160], desc[:400], cat))
                    got += 1
                    if got >= n_per_cat:
                        break
        except Exception as e:
            log(f"  seed {cat} error: {e}")
        log(f"  seeds[{cat}]: {got}")
    random.shuffle(seeds)
    return seeds


def build_prompt(title, desc):
    persona = random.choice(PERSONAS)
    length  = random.choice(LENGTHS)
    style   = random.choice(STYLES)
    stars   = random.choice(STARS)
    ctx = f"Product: {title}." + (f" Details: {desc}" if desc else "")
    user = (f"Write a realistic online customer review {length}, as if you are {persona}, "
            f"giving {stars} out of 5 stars. Write it {style}. Sound like a genuine human "
            f"shopper — mention specifics, do not be generic, and do NOT include the star "
            f"rating, a title, or any preamble. Just output the review text.\n\n{ctx}")
    return user, persona, style, stars


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (PowerShell: $env:OPENAI_API_KEY = 'sk-...').")
    from openai import OpenAI
    client = OpenAI()

    # pre-flight: one tiny call validates key / billing / model / network up front
    try:
        client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=5,
            messages=[{"role": "user", "content": "Reply with: ok"}])
        log("API key OK.")
    except Exception as e:
        sys.exit(f"OpenAI API check failed: {e}\n"
                 "Fix: verify your key is correct, that you've added billing/credits at "
                 "platform.openai.com (Settings -> Billing), and your network is up.")

    # resume: count existing rows
    done = 0
    new_file = not os.path.exists(OUT)
    if not new_file:
        with open(OUT, encoding="utf-8") as f:
            done = max(0, sum(1 for _ in f) - 1)
    log(f"already have {done} reviews; target {TOTAL}")
    if done >= TOTAL:
        log("target already met."); return

    log("collecting product seeds ...")
    seeds = collect_seeds()
    if not seeds:
        sys.exit("No product seeds collected (network?).")
    log(f"{len(seeds)} seeds.\nEstimated cost: a few dollars (mostly gpt-4o-mini).")

    weights = [w for *_, w in MODEL_CHOICES]
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["text", "domain", "model", "rating", "persona", "style"])

    made, i, fails = 0, 0, 0
    target_new = TOTAL - done
    while made < target_new:
        title, desc, cat = seeds[(done + i) % len(seeds)]
        i += 1
        model, temp, _ = random.choices(MODEL_CHOICES, weights=weights, k=1)[0]
        user, persona, style, stars = build_prompt(title, desc)
        try:
            r = client.chat.completions.create(
                model=model, temperature=temp, max_tokens=320,
                messages=[{"role": "system", "content": "You write realistic online product reviews."},
                          {"role": "user", "content": user}],
            )
            text = (r.choices[0].message.content or "").strip().replace("\n", " ")
            fails = 0
        except Exception as e:
            fails += 1
            log(f"  api error ({model}): {e}; backing off 5s (fail {fails}/10)")
            if fails >= 10:
                f.flush(); f.close()
                sys.exit(f"Stopping after 10 consecutive API errors. "
                         f"Saved {done + made} reviews to {OUT}.")
            time.sleep(5)
            continue
        if len(text.split()) < 4:
            continue
        w.writerow([text, "products", model, stars, persona, style])
        made += 1
        if made % 25 == 0:
            f.flush()
        if made % 200 == 0:
            log(f"  generated {made}/{target_new} (total {done + made})")

    f.flush()
    f.close()
    log(f"done — wrote {done + made} total to {OUT}")


if __name__ == "__main__":
    main()
