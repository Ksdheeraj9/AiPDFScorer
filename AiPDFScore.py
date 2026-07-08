pip install pypdf
pip install PyPDF2
import re
import math
import statistics
from collections import Counter
from PyPDF2 import PdfReader # Changed from pypdf to PyPDF2
from PyPDF2.errors import PdfReadError
import os


GENERIC_PHRASES = [
    "in conclusion", "overall", "moreover", "furthermore", "it is important to note",
    "this highlights", "in today's world", "on the other hand", "in summary",
    "additionally", "therefore", "thus", "ultimately", "significantly"
]


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if not os.path.isfile(pdf_path):
        raise ValueError(f"Path is not a file: {pdf_path}")

    with open(pdf_path, "rb") as f:
        header = f.read(8)

    # Real PDFs start with %PDF-
    if not header.startswith(b"%PDF-"):
        with open(pdf_path, "rb") as f:
            preview = f.read(120)
        raise ValueError(
            f"Invalid PDF header: {header!r}. "
            f"File looks non-PDF (preview={preview!r})."
        )

    try:
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            chunks.append(txt)
        text = "\n".join(chunks).strip()
    except PdfReadError as e:
        raise ValueError(f"Could not parse PDF: {e}")

    if not text:
        raise ValueError(
            "PDF parsed but no extractable text was found. "
            "It may be an image/scanned PDF (OCR required)."
        )

    return text


def split_sentences(text: str):
    # basic sentence split
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ai_likelihood_score(text: str):
    if not text or len(text) < 300:
        return {
            "score_percent": 0.0,
            "note": "Text too short for stable estimate.",
            "features": {}
        }

    sents = split_sentences(text)
    tokens = tokenize(text)
    n_tokens = len(tokens)

    # --- Feature 1: burstiness (sentence length variance) ---
    sent_lengths = [len(tokenize(s)) for s in sents if len(tokenize(s)) > 0]
    if len(sent_lengths) > 1:
        mean_len = statistics.mean(sent_lengths)
        stdev_len = statistics.pstdev(sent_lengths)
        cv = stdev_len / mean_len if mean_len > 0 else 0  # coefficient of variation
    else:
        cv = 0

    # lower variation can be AI-ish in many cases
    f_burstiness = clamp((0.35 - cv) / 0.35)  # cv <= 0.35 pushes score up

    # --- Feature 2: repetition (repeated trigrams) ---
    trigrams = list(zip(tokens, tokens[1:], tokens[2:])) if n_tokens >= 3 else []
    tri_counts = Counter(trigrams)
    repeated_tri = sum(c - 1 for c in tri_counts.values() if c > 1)
    rep_ratio = repeated_tri / max(1, len(trigrams))
    f_repetition = clamp(rep_ratio * 8)  # scale up

    # --- Feature 3: lexical diversity (very low diversity can be AI-ish) ---
    unique_ratio = len(set(tokens)) / max(1, n_tokens)
    f_lexical = clamp((0.45 - unique_ratio) / 0.25)  # unique_ratio < ~0.45 increases score

    # --- Feature 4: generic phrase frequency ---
    low = text.lower()
    generic_hits = sum(low.count(p) for p in GENERIC_PHRASES)
    generic_per_1k = generic_hits / max(1, (n_tokens / 1000))
    f_generic = clamp(generic_per_1k / 8)  # >=8 hits per 1k tokens is suspicious

    # --- Feature 5: punctuation entropy (overly regular punctuation can be AI-ish) ---
    punct = re.findall(r"[.,;:!?]", text)
    if punct:
        p_counts = Counter(punct)
        total = sum(p_counts.values())
        probs = [c / total for c in p_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        # low entropy -> repetitive punctuation usage
        f_punct = clamp((2.0 - entropy) / 2.0)
    else:
        f_punct = 0.0

    # Weighted blend (tune as needed)
    weights = {
        "burstiness": 0.30,
        "repetition": 0.25,
        "lexical": 0.20,
        "generic": 0.15,
        "punct": 0.10
    }

    raw = (
        f_burstiness * weights["burstiness"] +
        f_repetition * weights["repetition"] +
        f_lexical * weights["lexical"] +
        f_generic * weights["generic"] +
        f_punct * weights["punct"]
    )

    score_percent = round(clamp(raw) * 100, 2)

    return {
        "score_percent": score_percent,
        "note": "Heuristic estimate only (not proof).",
        "features": {
            "burstiness_low_variation": round(f_burstiness, 4),
            "repetition_trigrams": round(f_repetition, 4),
            "lexical_low_diversity": round(f_lexical, 4),
            "generic_phrase_density": round(f_generic, 4),
            "punctuation_regularness": round(f_punct, 4),
            "token_count": n_tokens,
            "sentence_count": len(sents)
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate AI-likelihood from PDF text.")
    parser.add_argument("--pdf_path", type=str, default=None, help="Path to PDF file")

    # Ignore unknown args injected by ipykernel (e.g., -f)
    args, _ = parser.parse_known_args()

    # Define a default sample PDF path for demonstration
    default_pdf_path = "/content/sample_data/samplefile.pdf"

    actual_pdf_path = None
    if args.pdf_path and os.path.exists(args.pdf_path) and os.path.isfile(args.pdf_path):
        actual_pdf_path = args.pdf_path
    elif os.path.exists(default_pdf_path) and os.path.isfile(default_pdf_path):
        print(f"No valid pdf_path provided, I'm Using default sample PDF: {default_pdf_path}")
        actual_pdf_path = default_pdf_path
    else:
        print("No valid PDF path could be determined for processing.")
        print(f"Please provide a valid PDF path or ensure the default path '{default_pdf_path}' exists.")

    if actual_pdf_path:
        try:
            text = extract_text_from_pdf(actual_pdf_path)
            result = ai_likelihood_score(text)
            print(result)
        except Exception as e:
            print(f"An error occurred while processing '{actual_pdf_path}': {e}")
            print("Tip: verify the file starts with %PDF- and is not JSON/HTML.")
    else:
        print("In notebook, you can also call the function directly like:")
        print(f"text = extract_text_from_pdf('{default_pdf_path}')")
        print("print(ai_likelihood_score(text))")
