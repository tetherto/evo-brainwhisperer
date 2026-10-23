import torch
from torch import Tensor, nn
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from brainwhisperer.config.schema import BWEncoderConfig


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
            attention_dropout=cfg.dropout,
            activation_dropout=cfg.dropout,
            activation_function="gelu",
            attn_implementation="sdpa",
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(layer_config) for _ in range(cfg.n_layers)]
        )
        # Replicating HF's weights initialization
        self.layers.apply(_hf_init_weights)
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Temporal position encoding added to the last d_model//2 dims (the embedder
        # occupies the first d_model//2 dims with the session encoding).
        temporal_pe = (
            torch.randn(cfg.max_source_pos, cfg.d_model // 2) * cfg.temporal_pe_std
        )
        self.register_buffer("pos_encoding", temporal_pe)

        if cfg.rope:
            # TODO: T5-style relative position bias (rope=False in all logged runs)
            pass

    def forward(self, x: Tensor) -> BaseModelOutput:
        """x: (B, 1500, d_model)"""
        _, T, D = x.shape

        # Add temporal position encoding to last D//2 dims (frozen random; see __init__)
        half = D // 2
        pos_enc = self.pos_encoding[:T]  # (T, D//2)
        # Adding temporal PE on the last half
        x = torch.cat([x[:, :, :half], x[:, :, half:] + pos_enc.unsqueeze(0)], dim=-1)

        # Ensuring contiguous memory layout for downstream dropout calls.
        x = x.transpose(1, 2).contiguous().transpose(1, 2)

        x = self.dropout(x)

        windowed_mask = self._windowed_mask(T, x.device)

        all_hidden = (x,)
        for i, layer in enumerate(self.layers):
            # Forcing HF's LayerDrop check during training
            if self.training:
                torch.rand([], device=x.device)
            mask = windowed_mask if i <= self.last_phoneme_layer else None
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
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask.masked_fill_(blocked, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def _hf_init_weights(module: nn.Module, init_std=0.02) -> None:
    """Replicate transformers WhisperPreTrainedModel._init_weights (init_std=0.02)."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        module.weight.data.normal_(mean=0.0, std=init_std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
