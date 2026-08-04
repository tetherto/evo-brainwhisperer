import torch
from torch import Tensor, nn
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from brainwhisperer.config.schema import BWEncoderConfig
from brainwhisperer.models.positional_encoding import PositionalEncoding


class BWEncoder(nn.Module):
    """
    Modified Whisper encoder with the two main contribution of the study.
    The implementation replaces the Whisper encoder stack without subclassing it.

    Input:  (B, 1500, d_model) [i.e., the embedder output].
    Output: BaseModelOutput with:
        last_hidden_state: (B, 1500, d_model)
        hidden_states:     tuple of `n_layers + 1` tensors, index 0 = encoder input,
                           index k = output of layer k-1.

    The first layers (up to `last_phoneme_layer`) use windowed attention
    where each position attends only `± attn_window_size//2` tokens.
    Remaining layers use full attention.
    The hidden state from the `last_phoneme_layer + 1` layer is read for the CTC phoneme head.
    """

    def __init__(self, cfg: BWEncoderConfig):
        super().__init__()
        self.last_phoneme_layer = cfg.last_phoneme_layer
        self.attn_window_size = cfg.attn_window_size

        # Build WhisperEncoderLayer instances from a minimal WhisperConfig.
        layer_config = WhisperConfig(
            d_model=cfg.d_model,
            encoder_attention_heads=cfg.n_heads,
            encoder_ffn_dim=cfg.d_ff,
            dropout=cfg.dropout,
            attention_dropout=0.0,
            activation_function="gelu",
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(layer_config) for _ in range(cfg.n_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Sinusoidal temporal encoding added to last d_model//2 dims,
        # while session encoding has been added to first d_model//2 dims by the embedder.
        self.pos_encoding = PositionalEncoding(cfg.d_model, max_len=_MAX_SEQ_LEN)

        if cfg.rope:
            # TODO: T5-style relative position bias (rope=False in all logged runs)
            pass

    def forward(self, x: Tensor) -> BaseModelOutput:
        """x: (B, 1500, d_model)"""
        _, T, D = x.shape

        # Add sinusoidal sequence-position encoding to last D//2 dims
        positions = torch.arange(T, device=x.device)  # (T,)
        pos_enc = self.pos_encoding(positions)  # (T, D//2)
        half = D // 2
        x = torch.cat([x[:, :, :half], x[:, :, half:] + pos_enc.unsqueeze(0)], dim=-1)

        x = self.dropout(x)

        windowed_mask = self._windowed_mask(T, x.device)

        all_hidden = (x,)
        for i, layer in enumerate(self.layers):
            mask = windowed_mask if i < self.last_phoneme_layer else None
            x = layer(x, attention_mask=mask, layer_head_mask=None)[0]
            all_hidden = all_hidden + (x,)

        x = self.layer_norm(x)
        # Replace final entry with the normed version so last_hidden_state is normed
        all_hidden = all_hidden[:-1] + (x,)

        return BaseModelOutput(last_hidden_state=x, hidden_states=all_hidden)

    def _windowed_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Additive attention mask of shape (1, 1, seq_len, seq_len).
        Positions outside ±attn_window_size//2 receive -1e9 (effectively -inf).
        """
        radius = self.attn_window_size // 2
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        blocked = (i - j).abs() > radius
        mask = blocked.float() * -1e9
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


_MAX_SEQ_LEN = 10000
