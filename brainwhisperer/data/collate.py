import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from brainwhisperer.data.batch import NeuralBatch


class NeuralCollator:
    """
    Collates per-sample dicts into a NeuralBatch.

    Tokenizer is injected so dataset classes stay model-agnostic: swapping the
    underlying language model only requires passing a different tokenizer here.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_label_len: int = 448):
        self.tokenizer = tokenizer
        self.max_label_len = max_label_len

    def __call__(self, batch: list[dict]) -> NeuralBatch:
        sbj_idx = torch.tensor([s["sbj_idx"] for s in batch], dtype=torch.long)

        feats = [s["neural_features"] for s in batch]  # each (T_i, C)
        neural_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        neural_features = pad_sequence(
            feats, batch_first=True, padding_value=0.0
        )  # (B, T_max, C)

        day = torch.tensor(
            [
                int(s["day"].item())
                if isinstance(s["day"], torch.Tensor)
                else int(s["day"])
                for s in batch
            ],
            dtype=torch.long,
        )

        tok_out = self.tokenizer(
            [s["sentence_label"] for s in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_label_len,
        )

        # Strip the leading BOS token and replace padding with -100,
        # so that CrossEntropyLoss will ignore those positions.
        # Pad token and EOS IDs are equal in Whisper, so masking
        # needs to be done via the attention mask to keep the trailing EOS the decoder must learn to emit.
        labels = tok_out["input_ids"][:, 1:]
        pad_mask = tok_out["attention_mask"][:, 1:] == 0
        labels = labels.masked_fill(pad_mask, -100)

        # Collate phoneme targets if every sample in the batch provides them.
        phone_seq = phone_seq_len = None
        if all("seq_class_ids" in s for s in batch):
            seqs = [s["seq_class_ids"] for s in batch]
            phone_seq_len = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            phone_seq = pad_sequence(seqs, batch_first=True, padding_value=0)

        return NeuralBatch(
            signals=neural_features.permute(0, 2, 1),  # (B, T_max, C) → (B, C, T_max)
            labels=labels,
            sbj_idx=sbj_idx,
            day_idx=day,
            lengths=neural_lengths,
            phone_seq=phone_seq,
            phone_seq_len=phone_seq_len,
        )
