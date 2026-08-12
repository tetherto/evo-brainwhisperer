"""Text normalisation used for WER / CER computation."""

import json
from pathlib import Path

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

_SPELLING_MAP = Path(__file__).parent / "english.json"


def build_normalizer() -> EnglishTextNormalizer:
    # "Whisper's English normaliser with custom spelling map
    with open(_SPELLING_MAP) as f:
        return EnglishTextNormalizer(json.load(f))
