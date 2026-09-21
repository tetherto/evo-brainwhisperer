"""WER, CER, and PER as pure functions over lists."""

from torchaudio.functional import edit_distance as _levenshtein


def word_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """Macro-averaged WER over a list of (hypothesis, reference) pairs."""
    total_edit = total_len = 0
    for hyp, ref in zip(hypotheses, references):
        total_edit += _levenshtein(hyp.split(), ref.split())
        total_len += len(ref.split())
    return total_edit / total_len if total_len else 0.0


def char_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """Macro-averaged CER over a list of (hypothesis, reference) pairs."""
    total_edit = total_len = 0
    for hyp, ref in zip(hypotheses, references):
        total_edit += _levenshtein(list(hyp), list(ref))
        total_len += len(ref)
    return total_edit / total_len if total_len else 0.0


def phoneme_error_rate(
    hypotheses: list[list[int]], references: list[list[int]]
) -> float:
    """PER over CTC-decoded phoneme index sequences (blank=0 already removed)."""
    total_edit = total_len = 0
    for hyp, ref in zip(hypotheses, references):
        total_edit += _levenshtein(hyp, ref)
        total_len += len(ref)
    return total_edit / total_len if total_len else 0.0
