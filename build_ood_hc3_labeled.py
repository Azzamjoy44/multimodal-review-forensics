"""
build_ood_hc3_labeled.py — out-of-distribution labeled browse set for the Fake Review Detection
section, so the UI can toggle in-distribution (held-out Amazon) vs OOD. Uses HC3
(Hello-SimpleAI/HC3): human vs ChatGPT answers — a corpus the unified detector was NEVER trained
on, differing in BOTH generator (ChatGPT) and text type (Q&A, not reviews). Good zero-shot
performance here = a transferable machine-vs-human signal, not a review/generator fingerprint.
Output: data/ood_hc3_labeled.csv  (1 = AI/ChatGPT fake, 0 = human genuine)
RUN: python build_ood_hc3_labeled.py
"""
import os
import csv
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
PER_CLASS = 400


def main():
    spec = importlib.util.spec_from_file_location("ext", os.path.join(HERE, "evaluate_cgor_external.py"))
    ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    texts, y = ext.load_hc3(PER_CLASS)          # balanced human/machine, deduped, min-words filtered
    out = os.path.join(DATA, "ood_hc3_labeled.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "review_text", "ground_truth_label", "source"])
        for i, (t, lab) in enumerate(zip(texts, y)):
            w.writerow([f"hc3_{i}", t, int(lab), "hc3_chatgpt" if lab == 1 else "hc3_human"])
    n_fake = int(sum(y))
    print(f"wrote data/ood_hc3_labeled.csv: {len(texts)} reviews ({n_fake} AI / {len(texts)-n_fake} human)")


if __name__ == "__main__":
    main()
