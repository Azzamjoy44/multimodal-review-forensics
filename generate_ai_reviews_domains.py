"""
generate_ai_reviews_domains.py — Route B, domain-broadening (OpenAI API).

Generates machine-written reviews for the NON-product review domains so the fake
detector covers the same review types as the review detector (not just products):
    movies, restaurants, apps, books, games
Each domain uses a domain-specific prompt + a random subject (genre/cuisine/category)
+ varied persona / length / style / rating, so the CG is topic-diverse and realistic.
Pairs (later, in build_broadened_cgor.py) against the HUMAN reviews we already have
for each domain (IMDB, Yelp, sealuzh apps, Amazon books, Steam).

SETUP
    pip install openai
    PowerShell:  $env:OPENAI_API_KEY = "sk-..."
RUN (resumable per-domain; re-run to top up)
    python generate_ai_reviews_domains.py
    python generate_ai_reviews_domains.py 2000     # optional: per-domain target

OUTPUT: data/ai_generated_reviews_domains.csv  (text, domain, model, rating, persona, style, subject)
"""

import os
import sys
import csv
import random
import time

OUT = os.path.join(os.path.dirname(__file__), "data", "ai_generated_reviews_domains.csv")
PER_DOMAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
SEED = 42
random.seed(SEED)

MODEL_CHOICES = [  # (model, temperature, weight)
    ("gpt-4o-mini", 0.7, 0.45), ("gpt-4o-mini", 1.0, 0.25),
    ("gpt-4o-mini", 0.4, 0.15), ("gpt-3.5-turbo", 0.8, 0.10), ("gpt-4o", 0.7, 0.05),
]
PERSONAS = ["a busy parent","a college student","a retiree","a casual fan","a hardcore enthusiast",
            "a first-timer","a longtime regular","a skeptic","a bargain hunter","a critic at heart"]
LENGTHS = ["in just one short sentence","in one or two sentences","in a short paragraph",
           "in a detailed paragraph or two"]
STYLES = ["polished and well-written","casual and conversational",
          "quick and informal, all lowercase with a couple of typos like a phone review",
          "enthusiastic and full of energy","measured, balanced and a little critical","blunt and to the point"]
STARS = [5,5,5,5,4,4,4,3,2,1,1]

# domain -> (verb phrase, list of subjects)
DOMAINS = {
    "movies":      ("you watched", ["a sci-fi movie","a romantic comedy","a horror film","an action blockbuster",
                                    "a slow indie drama","an animated family movie","a documentary","a psychological thriller",
                                    "a superhero movie","a foreign-language film"]),
    "restaurants": ("you visited", ["an Italian restaurant","a sushi spot","a taco truck","a fine-dining steakhouse",
                                    "a vegan cafe","a burger joint","a Thai restaurant","a brunch place","a pizzeria",
                                    "a diner","a ramen shop","a food truck"]),
    "apps":        ("you've been using", ["a budgeting app","a fitness-tracking app","a photo-editing app",
                                          "a language-learning app","a meditation app","a food-delivery app",
                                          "a note-taking app","a ride-share app","a dating app","a streaming app"]),
    "books":       ("you read", ["a mystery novel","a fantasy epic","a self-help book","a historical romance",
                                 "a sci-fi novel","a memoir","a literary thriller","a cookbook","a young-adult novel",
                                 "a non-fiction history book"]),
    "games":       ("you played", ["an open-world RPG","an indie platformer","a first-person shooter",
                                    "a cozy farming sim","a roguelike","a racing game","a puzzle game",
                                    "a survival-horror game","a strategy game","a fighting game"]),
}


def log(m):
    print(m, flush=True)


def build_prompt(domain):
    verb, subjects = DOMAINS[domain]
    subject = random.choice(subjects)
    persona = random.choice(PERSONAS); length = random.choice(LENGTHS)
    style = random.choice(STYLES); stars = random.choice(STARS)
    user = (f"Write a realistic online review of {subject} {verb}, as if you are {persona}, "
            f"giving {stars} out of 5 stars. Write it {style}. Invent a plausible specific "
            f"title/name and concrete details, sound like a genuine human - do NOT include the "
            f"star rating, a header, or any preamble. Just output the review text.")
    return user, persona, style, stars, subject


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (PowerShell: $env:OPENAI_API_KEY = 'sk-...').")
    from openai import OpenAI
    client = OpenAI()
    try:
        client.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
            messages=[{"role": "user", "content": "Reply with: ok"}])
        log("API key OK.")
    except Exception as e:
        sys.exit(f"OpenAI API check failed: {e}\nVerify key + billing at platform.openai.com.")

    # resume: per-domain counts already present
    have = {d: 0 for d in DOMAINS}
    new_file = not os.path.exists(OUT)
    if not new_file:
        import pandas as pd
        try:
            dd = pd.read_csv(OUT)
            for d, c in dd["domain"].value_counts().items():
                if d in have:
                    have[d] = int(c)
        except Exception:
            pass
    log(f"existing per-domain: {have} | target {PER_DOMAIN} each")

    weights = [w for *_, w in MODEL_CHOICES]
    f = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(["text", "domain", "model", "rating", "persona", "style", "subject"])

    for domain in DOMAINS:
        need = PER_DOMAIN - have[domain]
        if need <= 0:
            log(f"[{domain}] already has {have[domain]}"); continue
        log(f"[{domain}] generating {need} ...")
        made, fails = 0, 0
        while made < need:
            model, temp, _ = random.choices(MODEL_CHOICES, weights=weights, k=1)[0]
            user, persona, style, stars, subject = build_prompt(domain)
            try:
                r = client.chat.completions.create(
                    model=model, temperature=temp, max_tokens=320,
                    messages=[{"role": "system", "content": "You write realistic online reviews."},
                              {"role": "user", "content": user}])
                text = (r.choices[0].message.content or "").strip().replace("\n", " ")
                fails = 0
            except Exception as e:
                fails += 1
                log(f"  api error ({model}): {e} (fail {fails}/10)")
                if fails >= 10:
                    f.flush(); f.close()
                    sys.exit(f"Stopping after 10 consecutive API errors. Saved progress to {OUT}.")
                time.sleep(5); continue
            if len(text.split()) < 4:
                continue
            w.writerow([text, domain, model, stars, persona, style, subject])
            made += 1
            if made % 25 == 0:
                f.flush()
            if made % 200 == 0:
                log(f"  {domain}: {made}/{need}")
        f.flush()
        log(f"[{domain}] done (+{made})")

    f.close()
    log(f"finished -> {OUT}")


if __name__ == "__main__":
    main()
