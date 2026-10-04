# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
make_presentation.py
--------------------
Generates `thesis_defense.pptx` — a tight, ~5-minute, 7-slide defense deck for
the diploma project "A Multi-Modal Adversarial Network for Real-Time Review
Forensics".

Run:
    pip install python-pptx
    python make_presentation.py

All numbers below are the verified results from the thesis (do not edit the
figures unless the underlying result changed). Easy-to-edit config at the top.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ------------------------------------------------------------------ config ---
import sys
STUDENT_NAME = "Azzam Zafar"                 # <-- edit if needed
UNIVERSITY   = "Bachelor's Diploma Project  ·  University POLITEHNICA of Bucharest  ·  FILS"
OUTPUT       = sys.argv[1] if len(sys.argv) > 1 else "thesis_defense.pptx"
FONT         = "Segoe UI"                     # clean sans-serif (Windows 11)

# ------------------------------------------------------------------ colours --
NAVY   = RGBColor(0x1E, 0x3A, 0x5F)          # accent / titles / big figures
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARK   = RGBColor(0x22, 0x2B, 0x35)          # body text
GRAY   = RGBColor(0x55, 0x5F, 0x6B)          # secondary text
LIGHT  = RGBColor(0xEC, 0xF1, 0xF7)          # box fill
GREEN  = RGBColor(0x1F, 0x7A, 0x3D)          # positive figure accent

# ------------------------------------------------------------------ helpers --
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def bg(s, color):
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = color


def text(s, txt, left, top, width, height, size, color,
         bold=False, italic=False, align=PP_ALIGN.LEFT, font=FONT,
         anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = txt
    f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic
    f.name = font; f.color.rgb = color
    return box


def bullets(s, items, left, top, width, height, size, color,
            font=FONT, space=14, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run(); r.text = it
        f = r.font; f.size = Pt(size); f.name = font; f.color.rgb = color
    return box


def rule(s, left, top, width, color=NAVY, height=0.06):
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                             Inches(left), Inches(top), Inches(width), Inches(height))
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background(); shp.shadow.inherit = False
    return shp


def title_block(s, txt):
    """Standard content-slide title + navy underline rule."""
    text(s, txt, 0.7, 0.45, 12.0, 1.0, 30, NAVY, bold=True)
    rule(s, 0.72, 1.45, 2.2)


def channel_box(s, head, body, left, top, width=3.74, height=3.7):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(left), Inches(top), Inches(width), Inches(height))
    shp.fill.solid(); shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = NAVY; shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.28); tf.margin_right = Inches(0.28)
    tf.margin_top = Inches(0.28)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = head
    r.font.size = Pt(19); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = NAVY
    p.space_after = Pt(10)
    for line in body:
        bp = tf.add_paragraph(); bp.alignment = PP_ALIGN.LEFT; bp.space_after = Pt(8)
        br = bp.add_run(); br.text = line
        br.font.size = Pt(13.5); br.font.name = FONT; br.font.color.rgb = DARK
    return shp


def stat(s, number, label, left, top, width=5.55, num_color=NAVY,
         num_size=46, label_size=14, size=None):
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(2.0))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = number
    r.font.size = Pt(num_size); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = num_color
    lp = tf.add_paragraph(); lp.alignment = PP_ALIGN.CENTER
    lr = lp.add_run(); lr.text = label
    lr.font.size = Pt(label_size); lr.font.name = FONT; lr.font.color.rgb = GRAY
    if size:
        sp = tf.add_paragraph(); sp.alignment = PP_ALIGN.CENTER; sp.space_before = Pt(2)
        sr = sp.add_run(); sr.text = size
        sr.font.size = Pt(label_size); sr.font.italic = True; sr.font.name = FONT; sr.font.color.rgb = NAVY
    return box


def kv_block(s, pairs, left, top, width, height,
             label_size=13.5, body_size=13.5, space=10):
    """Each pair = (bold navy lead label, normal dark remainder), one per line."""
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame; tf.word_wrap = True
    for i, (lab, rest) in enumerate(pairs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r1 = p.add_run(); r1.text = lab
        r1.font.size = Pt(label_size); r1.font.bold = True
        r1.font.name = FONT; r1.font.color.rgb = NAVY
        r2 = p.add_run(); r2.text = rest
        r2.font.size = Pt(body_size); r2.font.name = FONT; r2.font.color.rgb = DARK
    return box


# ============================================================== SLIDE 1 =====
s = slide(); bg(s, NAVY)
rule(s, 0.0, 0.0, 13.333, color=RGBColor(0x14, 0x29, 0x45), height=0.18)
text(s, "A Multi-Modal Adversarial Network for\nReal-Time Review Forensics",
     1.0, 2.05, 11.33, 2.2, 40, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "Detecting fake reviews across Amazon & Yelp — from text, behaviour, and network signals",
     1.0, 4.05, 11.33, 0.8, 17, RGBColor(0xCF, 0xDD, 0xEC), italic=True, align=PP_ALIGN.CENTER)
text(s, STUDENT_NAME, 1.0, 5.25, 11.33, 0.6, 20, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, UNIVERSITY, 1.0, 5.95, 11.33, 0.6, 14, RGBColor(0xB8, 0xCA, 0xDD), align=PP_ALIGN.CENTER)
notes(s, "Good morning. My diploma project is a real-time system for review "
         "forensics: detecting fake reviews on Amazon and Yelp. I'll cover the "
         "problem, my multi-modal approach, the key results, and the live "
         "application I built.")

# ============================================================== SLIDE 2 =====
s = slide(); bg(s, WHITE)
title_block(s, "The Problem: Fake Reviews Are Harder to Spot Than Ever")
bullets(s, [
    "AI now writes fluent, convincing fake reviews at scale",
    "Fraud is often coordinated — rings of reviewers, not lone actors",
    "No single signal is enough: a coordinated ring can post normal-looking reviews, and a lone faker shows no group pattern",
], 0.9, 2.1, 11.4, 3.0, 22, DARK, space=20)
text(s, "→  Different kinds of fake need different detectors — so I built one for each.",
     0.9, 5.45, 11.7, 0.9, 20, NAVY, bold=True)
notes(s, "Fake reviews mislead buyers, and they are harder to catch than ever. "
         "AI now writes fluent fakes, and fraud is often coordinated rather than "
         "lone actors. The catch is that no single signal is enough: a "
         "coordinated ring can write perfectly normal-looking reviews, so the text "
         "alone won't flag it, while a lone faker forms no suspicious group, so the "
         "connections alone won't flag it either. So different kinds of fake call "
         "for different detectors — I built one for "
         "each, rather than one model that does everything.")

# ============================================================== SLIDE 3 =====
s = slide(); bg(s, WHITE)
title_block(s, "Approach: Three Independent, Specialised Detectors")
channel_box(s, "TEXT", [
    "What the review says.",
    "A unified detector that catches both AI-generated and human-written deceptive reviews.",
], 0.7, 1.9)
channel_box(s, "BEHAVIOUR", [
    "How the reviewer acts.",
    "Fuses the text with 17 reviewer / business behavioural features (Yelp).",
], 4.80, 1.9)
channel_box(s, "RELATIONS", [
    "How reviewers connect.",
    "A graph neural network surfaces coordinated rings that per-review models cannot see.",
], 8.90, 1.9)
text(s, "Each text detector = a 10-model ensemble: 8 classical ML (KNN, RF, DT, LR, NB, "
        "XGBoost, AdaBoost, MLP) + BiLSTM + DistilBERT.  Plus a sentiment analyser and a review gate.",
     0.7, 5.85, 11.93, 0.9, 14, GRAY, align=PP_ALIGN.CENTER)
text(s, "Three independent detectors — text · text + behaviour · reviewer graph — each specialised for a different kind of fake (applied where it fits, not merged into one score).",
     0.7, 6.5, 11.93, 0.7, 13, NAVY, bold=True, italic=True, align=PP_ALIGN.CENTER)
notes(s, "My approach is multi-modal at the system level — three separate, "
         "specialised detectors, not one model, and they run independently rather "
         "than feeding each other. A text detector catches AI-generated and "
         "human-written fakes from wording — it serves the Amazon section and the "
         "upload box. A second model, for Yelp, fuses the text with seventeen "
         "behavioural features. And a graph network reads how reviewers connect, to "
         "expose coordinated rings, offline. Each is applied where it fits, not "
         "merged into a single verdict. Each text detector is itself a ten-model "
         "ensemble — eight classical models plus a BiLSTM and DistilBERT.")

# ============================================================== SLIDE 4 — TRAINING =====
s = slide(); bg(s, WHITE)
title_block(s, "How the Models Are Trained")
rule(s, 6.35, 1.95, 0.018, color=RGBColor(0xC9, 0xD3, 0xDE), height=4.55)
# left column — common recipe
text(s, "Common recipe — every detector", 0.72, 1.85, 5.45, 0.5, 18, NAVY, bold=True)
bullets(s, [
    "•  10 models per detector: 8 TF-IDF classical (KNN…XGBoost), trained locally",
    "•  + a BiLSTM (frozen GloVe) and a fine-tuned DistilBERT",
    "•  Deep models trained on Colab GPU → exported to ONNX for fast CPU serving",
    "•  Leak-free train / val / test splits (grouped on Yelp, where reviewers repeat)",
], 0.72, 2.5, 5.5, 4.2, 14, DARK, space=14)
# right column — behind each result
text(s, "Training data per detector", 6.7, 1.85, 5.9, 0.5, 18, NAVY, bold=True)
kv_block(s, [
    ("Fake detector:  ", "132k reviews — AI-generated, human-deceptive & genuine; rare human class oversampled 15×"),
    ("Sentiment:  ", "500k reviews — 50k IMDB movies + 450k tweets (stop words removed)"),
    ("Review gate:  ", "599k texts — balanced reviews vs non-reviews, 24 sources (keeps stop words)"),
    ("Yelp (Model B):  ", "~679k Yelp reviews — text + 17 reviewer / business behavioural features"),
    ("Rings (GNN):  ", "Yelp graph (~679k reviews) + Amazon graph (~12k reviewers); GraphSAGE → rings"),
], 6.7, 2.5, 5.95, 4.5, label_size=13, body_size=13, space=11)
notes(s, "How the models are trained. Every detector is the same ten-model recipe "
         "— eight TF-IDF classical models locally, plus a BiLSTM and a fine-tuned "
         "DistilBERT trained on a Colab GPU and exported to ONNX — all on leak-free "
         "splits, grouped on Yelp so a reviewer can't span train and test. The fake "
         "detector learns from 131,768 reviews — 64,478 AI-generated, 1,636 "
         "human-written deceptive, and 65,654 genuine — with the rare human class "
         "oversampled fifteen times. Provenance: the AI-generated reviews come from "
         "Salminen et al.'s 2022 Kaggle 'Fake Reviews' dataset (a GPT-2 base), "
         "broadened with our own modern-LLM generations (Llama-3, Qwen, Zephyr, Yi, "
         "GPT-4o-mini) plus the Lyra / sutro / Kenshiii AI-review corpora; the "
         "human-deceptive reviews from the Ott (2011/2013) and Li (2014) "
         "deceptive-opinion-spam corpora — Turkers paid to write fakes; the genuine "
         "ones are real human reviews pooled across domains — Amazon, Yelp, IMDB, "
         "books, Steam and apps — matched to the fakes' domains so the model can't "
         "shortcut on topic. Sentiment uses 500,000 reviews: 50,000 IMDB movie "
         "reviews (Maas 2011) plus 450,000 tweets (Sentiment140, Go 2009). The "
         "review/non-review gate trains on 599,154 texts, balanced 50/50 across 24 "
         "sources — reviews (Amazon, IMDB, Yelp, Steam) versus non-reviews (product "
         "descriptions, Q&A, news, code, prose); unlike the others it keeps stop "
         "words, since first- and second-person pronouns are strong review cues. "
         "Model B fuses text with seventeen behavioural features over the full "
         "deduplicated Yelp set of 678,663 reviews (~13% fake) from four corpora — "
         "YelpZip, YelpNYC, YelpChi restaurants and hotels (Rayana & Akoglu 2015; "
         "Mukherjee 2013). The GNN trains on the same Yelp review graph (678,663 "
         "reviews, 302,821 reviewers, 6,168 businesses) plus a separate Amazon fraud "
         "graph of about 12,000 reviewers — the CARE-GNN Musical Instruments "
         "benchmark — around 9.5% fraudulent.")


# ============================================================== SLIDE 5 =====
s = slide(); bg(s, WHITE)
title_block(s, "Results")
# Row 1 — the three text detectors
stat(s, "92.7% / 73.8%",
     "ensemble per-class recall — AI / human (98% gen. spec.)",
     0.55, 2.0, width=3.95, num_size=27, label_size=11, size="tested on 13,179 reviews")
stat(s, "86.7%",
     "sentiment accuracy, cross-domain — best of 10: DistilBERT · worst KNN 57.8%",
     4.69, 2.0, width=3.95, num_color=GREEN, num_size=36, label_size=11, size="tested on 10,000 reviews")
stat(s, "99.5%",
     "review / non-review gate, macro-F1 — best of 10: DistilBERT · worst AdaBoost 82.4%",
     8.83, 2.0, width=3.95, num_color=GREEN, num_size=36, label_size=11, size="tested on 59,916 texts")
# Row 2 — Yelp Model B + the GNN rings (centred)
stat(s, "68 → 77%",
     "Yelp macro-F1: text-only → text + behaviour · 10-model ensemble",
     2.55, 4.45, width=3.95, num_size=36, label_size=11, size="tested on 17,891 reviews")
stat(s, "88% / 60%",
     "fraud inside GNN rings — Amazon / Yelp\nvs base rates ~9.5% / ~13% → 9× / 4.5× (GraphSAGE)",
     6.83, 4.45, width=3.95, num_size=34, label_size=11, size="Amazon ~12k · Yelp ~679k graphs")
text(s, "Ensemble figure where the detector is a 10-model ensemble (fake · Yelp); otherwise the best single model, weakest shown for range. GNN is a single GraphSAGE.",
     0.55, 6.78, 12.25, 0.5, 11, GRAY, italic=True, align=PP_ALIGN.CENTER)
notes(s, "These are the headline results — one per detector from slide 4, each on a "
         "held-out, leak-free test. Fake detector: 13,179 reviews (6,448 "
         "AI-generated, 164 human-written, 6,567 genuine); the ten-model ensemble "
         "catches 92.7% of AI and 73.8% of human fakes at 98% genuine specificity "
         "(best single DistilBERT 99.5 / 92.7). Sentiment: 86.7% accuracy on 10,000 "
         "balanced Amazon reviews, cross-domain (DistilBERT; weakest KNN 57.8%). "
         "Review gate: 99.5% macro-F1 on 59,916 texts (DistilBERT; weakest AdaBoost "
         "82.4%). Yelp Model B: ensemble macro-F1 rises from 68.4 text-only to 76.7 "
         "with behaviour, on 17,891 reviews. Rings: the GNN's extracted rings are "
         "88% fraud on Amazon and 60% on Yelp — about 9× and 4.5× their base rates "
         "(~9.5% and ~13%). Not on the slide but worth knowing: behaviour alone (a "
         "single XGBoost) reaches 75.9% on Yelp; the GNN classifies fraud at 76.3% "
         "(Yelp) / 90.9% (Amazon) with no text; and on robustness — all "
         "out-of-distribution — it generalises to unseen AI generators (mean 0.79, "
         "leave-one-generator-out) and unseen domains (mean 0.77), and flags only "
         "0.2% of genuine reviews at full 10-model consensus.")

# ============================================================== SLIDE 6 =====
s = slide(); bg(s, WHITE)
title_block(s, "A Live, Real-Time Web Application")
bullets(s, [
    "FastAPI backend + lightweight JavaScript frontend",
    "Models pre-loaded at startup — deep models (BiLSTM, DistilBERT) served via ONNX Runtime, the 8 sklearn via joblib; no PyTorch/TensorFlow at request time, runs on a modest machine",
    "Four sections: Fake Review Detection · Sentiment · Dashboard (a reviewer's history) · GNN Rings",
    "Analyze box: paste text, upload a CSV, or screenshots (image OCR) — plus a per-user “My Uploads” history",
    "Every input first passes a review / non-review gate, so only real reviews get scored",
], 0.9, 1.95, 11.7, 4.0, 18, DARK, space=12)
text(s, "(Live demo to follow)", 0.9, 6.5, 11.4, 0.6, 16, NAVY, bold=True, italic=True)
notes(s, "It is a working real-time application: a FastAPI backend, a lightweight "
         "JavaScript frontend, all models pre-loaded at startup — the two deep "
         "models per detector (BiLSTM and DistilBERT) served through ONNX Runtime, "
         "with no PyTorch or TensorFlow at request time, and the eight sklearn via "
         "joblib — so it runs on a modest machine. Four sections: fake detection, "
         "sentiment, a reviewer dashboard, and the GNN rings. The Analyze box takes "
         "text, CSVs, or screenshots via OCR — and every input first passes a "
         "ten-model review/non-review gate (DistilBERT at 99.5% macro-F1), so only "
         "real reviews get scored. It also saves each user's history. I'll show it "
         "live.")

# ============================================================== SLIDE 7 — ANALYZE DEMO =====
s = slide(); bg(s, WHITE)
title_block(s, "The Analyze Section — Live Demo")
_vid = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.57), Inches(1.75), Inches(8.2), Inches(4.6))
_vid.fill.solid(); _vid.fill.fore_color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
_vid.line.color.rgb = NAVY; _vid.line.width = Pt(1.25); _vid.shadow.inherit = False
_tf = _vid.text_frame; _tf.word_wrap = True; _tf.vertical_anchor = MSO_ANCHOR.MIDDLE
_p = _tf.paragraphs[0]; _p.alignment = PP_ALIGN.CENTER
_r = _p.add_run(); _r.text = "▶"
_r.font.size = Pt(54); _r.font.name = FONT; _r.font.color.rgb = WHITE
_h = _tf.add_paragraph(); _h.alignment = PP_ALIGN.CENTER
_hr = _h.add_run(); _hr.text = "insert the Analyze-section demo video here"
_hr.font.size = Pt(14); _hr.font.italic = True; _hr.font.name = FONT; _hr.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEC)
text(s, "Figure 6.4 — Analyze section (demo video)", 2.57, 6.5, 8.2, 0.45, 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
notes(s, "Here I play a short demo of the Analyze section (figure 6.4). [Describe "
         "what the clip shows: pasting a review or uploading a CSV / screenshot, and "
         "the per-model fake and sentiment verdicts with highlighted phrases.] This "
         "is the user-facing entry point: any text, CSV, or image is gated, scored "
         "by the unified fake detector and the sentiment ensemble, and saved to the "
         "user's My Uploads.")


# ============================================================== SLIDE 8 =====
s = slide(); bg(s, WHITE)
title_block(s, "Conclusion")
bullets(s, [
    "Built one real-time app combining three contributions:",
    "  •  a generalised text detector for both AI and human fakes",
    "  •  a behavioural fusion that breaks the text-only ceiling",
    "  •  a graph view that exposes coordinated rings",
], 0.9, 1.95, 11.4, 2.6, 20, DARK, space=12)
text(s, "Honest findings:  imbalance tricks (focal loss, contrastive learning) did not beat "
        "plain training, and the Yelp text ceiling is set by the labels, not the model.",
     0.9, 4.85, 11.4, 1.0, 16, GRAY, italic=True)
text(s, "Thank you — questions welcome.", 0.9, 6.3, 11.4, 0.7, 22, NAVY, bold=True)
notes(s, "In summary, I built a generalised text detector for both AI and human "
         "fakes, a behavioural fusion that breaks the text ceiling, and a graph "
         "view that exposes rings — all in one real-time app. I also report honest "
         "negatives: imbalance tricks didn't help, and the Yelp ceiling is set by "
         "the labels, not the model. Thank you — happy to take questions.")

# ------------------------------------------------------------------ save -----
prs.save(OUTPUT)
print(f"Wrote {OUTPUT} with {len(prs.slides)} slides.")
