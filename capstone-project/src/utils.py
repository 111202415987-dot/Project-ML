"""
utils.py
Fungsi utilitas bersama untuk preprocessing teks SMS.
Digunakan oleh notebook (EDA & modeling) maupun aplikasi Streamlit,
agar transformasi teks pada saat training dan saat inference konsisten.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Pastikan resource NLTK tersedia
for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(
            f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Pipeline preprocessing teks SMS:
    1. Case folding
    2. Cleaning (hapus angka, tanda baca, karakter non-alfabet)
    3. Tokenizing (split spasi)
    4. Stopword removal
    5. Stemming (PorterStemmer)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    filtered = [_STEMMER.stem(w) for w in words if w not in _STOPWORDS]
    return " ".join(filtered)


def count_message_stats(text: str) -> dict:
    """Statistik dasar sebuah pesan (dipakai untuk feature engineering & EDA)."""
    if not isinstance(text, str):
        text = ""
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "has_number": int(bool(re.search(r"\d", text))),
        "has_currency": int(bool(re.search(r"[$£€]|\bfree\b", text.lower()))),
        "exclamation_count": text.count("!"),
    }
