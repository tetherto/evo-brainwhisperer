"""Text normalisation used for WER / CER computation."""

import json
import re
from pathlib import Path

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

_SPELLING_MAP = Path(__file__).parent / "english.json"


def build_normalizer() -> EnglishTextNormalizer:
    # "Whisper's English normaliser with custom spelling map
    with open(_SPELLING_MAP) as f:
        return EnglishTextNormalizer(json.load(f))


def remove_punctuation(sentence: str) -> str:
    sentence = re.sub(r"[^a-zA-Z\- \']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = " ".join([word for word in sentence.split() if word != ""])

    return sentence
