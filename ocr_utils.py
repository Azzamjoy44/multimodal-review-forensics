"""
ocr_utils.py — extract review text from an uploaded image for the Dashboard "Analyze"
feature, using **Mistral** vision OCR.

Why Mistral (and not the old EasyOCR): EasyOCR transcribed *everything* in the screenshot
— reviewer name, star rating, date, "Verified Purchase", "Helpful", "report", "Translated
by Amazon", etc. — and fed all that chrome to the models, which confused them. It also
merged several stacked reviews into one blob. Mistral's vision model is *instructed* to
return ONLY each review's body text, with one entry per distinct review, so:
  * surrounding UI / metadata is stripped, and
  * an image with N reviews yields N separate reviews (each analysed independently).

`extract_reviews_from_image(image_bytes, mime)` -> list[str] of clean review bodies.

Requires the `mistralai` package and the MISTRAL_API_KEY environment variable.
Model is `pixtral-12b-2409` by default; override with MISTRAL_OCR_MODEL.
"""

import os
import json
import base64
import threading

_client = None
_client_lock = threading.Lock()

MODEL = os.environ.get("MISTRAL_OCR_MODEL", "pixtral-12b-2409")

_PROMPT = (
    "You are extracting customer REVIEW text from a screenshot of a product/business review "
    "page (e.g. Amazon, Yelp). Return ONLY the words the reviewer actually wrote.\n\n"
    "Return a JSON object of the exact form: {\"reviews\": [\"...\", \"...\"]}\n"
    "  - One array element per DISTINCT review visible in the image.\n"
    "  - Each element = ONLY that review's body text (the paragraph(s) the reviewer wrote), "
    "transcribed verbatim (do NOT paraphrase, summarise, translate, or fix spelling). If a "
    "review shows translated text (e.g. 'Translated from German by Amazon'), transcribe the "
    "translated body that is displayed.\n\n"
    "EXCLUDE everything that is interface, metadata, or a heading — NOT the reviewer's body prose:\n"
    "  - the review's bold title / headline (the short bold heading shown above the body, e.g. "
    "'Great radio', 'Nice gem!') — do NOT include it\n"
    "  - reviewer name / username / avatar\n"
    "  - star rating, number of stars, rating value\n"
    "  - date, location ('Reviewed in ... on ...')\n"
    "  - 'Verified Purchase', 'Colour Name: ...', 'Size: ...', variant labels\n"
    "  - 'Helpful', 'report', '... found this helpful', vote counts, button labels\n"
    "  - 'Translated from ... by Amazon', 'Translate review to English', 'See original', "
    "product titles, page headers.\n\n"
    "If there are no reviews in the image, return {\"reviews\": []}. "
    "Output ONLY the JSON object — no commentary, no markdown fences."
)


class OcrUnavailable(RuntimeError):
    pass


def ocr_available() -> bool:
    """True iff the mistralai SDK is importable AND an API key is configured."""
    import importlib.util
    return (importlib.util.find_spec("mistralai") is not None
            and bool(os.environ.get("MISTRAL_API_KEY")))


def _get_client():
    """Lazily build a single Mistral client (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                key = os.environ.get("MISTRAL_API_KEY")
                if not key:
                    raise OcrUnavailable(
                        "MISTRAL_API_KEY is not set. Set it in the environment "
                        "(PowerShell: $env:MISTRAL_API_KEY=\"...\") and restart the server."
                    )
                try:
                    from mistralai import Mistral
                except ImportError as e:
                    raise OcrUnavailable(
                        "The 'mistralai' package is not installed. Run `pip install mistralai`."
                    ) from e
                _client = Mistral(api_key=key)
    return _client


def _parse_reviews(content: str) -> list:
    """Pull the reviews list out of the model's JSON reply, defensively."""
    if not content:
        return []
    txt = content.strip()
    if txt.startswith("```"):                      # strip ```json ... ``` fences if present
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # last resort: grab the first {...} or [...] block
        for o, c in (("{", "}"), ("[", "]")):
            i, j = txt.find(o), txt.rfind(c)
            if i != -1 and j > i:
                try:
                    data = json.loads(txt[i:j + 1]); break
                except json.JSONDecodeError:
                    continue
        else:
            return []
    revs = data.get("reviews", []) if isinstance(data, dict) else data
    return [str(r).strip() for r in revs if isinstance(r, (str, int, float)) and str(r).strip()]


def extract_reviews_from_image(image_bytes: bytes, mime: str = "image/png") -> list:
    """Return a list of clean review bodies found in the image (chrome stripped,
    one entry per distinct review). Empty list if none found."""
    client = _get_client()
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.chat.complete(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            ],
        }],
        temperature=0,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):   # some SDK versions return content chunks
        content = "".join(getattr(c, "text", "") or (c.get("text", "") if isinstance(c, dict) else "")
                          for c in content)
    return _parse_reviews(content or "")
