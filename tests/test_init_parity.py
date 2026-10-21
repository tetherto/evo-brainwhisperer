"""
Train-time parity guards.

The model must reproduce the published runs' *initialization* and *regularization*,
not just their forward computation. Init and dropout only affect training dynamics:
with identical weights the eval-mode output is identical, so forward tests cannot
catch a change here.

Each assertion below pins a setting that once diverged from the published runs while
still giving a bit-identical forward for the same weights. "REF" in the messages means
the published runs' code.
"""

import torch
from torch import nn

from brainwhisperer.config.schema import BWEmbedderConfig, BWEncoderConfig
from brainwhisperer.models.embedder import BWEmbedder
from brainwhisperer.models.encoder import BWEncoder

_INIT_STD = 0.02  # whisper-tiny WhisperConfig.init_std; REFERENCE encoder gets it via post_init()


def _encoder() -> BWEncoder:
    return BWEncoder(
        BWEncoderConfig(
            d_model=384,
            n_heads=6,
            d_ff=1536,
            n_layers=6,
            dropout=0.1,
            last_phoneme_layer=3,
            attn_window_size=57,
            rope=False,
            max_source_pos=1500,
            temporal_pe_std=0.02,
        )
    )


def _embedder(rank: int = 16) -> BWEmbedder:
    return BWEmbedder(
        BWEmbedderConfig(
            num_features=512,
            d_model=384,
            embed_type="conv",
            kernel_size_1=7,
            stride_2=2,
            patch_size=5,
            encodings="sin",
            day_projections=True,
            day_proj_rank=rank,
            dropout=0.4,
            day_layer_dropout=0.2,
            session_dropout=0.1,
        ),
        num_days=45,
    )


class TestEncoderInitParity:
    """REFERENCE WhisperEncoder_ subclasses WhisperPreTrainedModel → post_init() runs HF
    _init_weights (std=0.02). Composing WhisperEncoderLayer directly skips post_init and would
    leave PyTorch's fan-in default (~0.03)."""

    def test_layer_weights_use_hf_init(self):
        enc = _encoder()
        for i, layer in enumerate(enc.layers):
            params = dict(layer.named_parameters())
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "fc1",
                "fc2",
            ):
                std = params[name + ".weight"].std().item()
                assert abs(std - _INIT_STD) < 0.004, (
                    f"encoder layer{i}.{name}.weight std={std:.4f}, expected HF init "
                    f"~{_INIT_STD}. PyTorch default (~0.03) means post_init was skipped."
                )

    def test_temporal_pe_is_frozen_random_not_sinusoidal(self):
        enc = _encoder()
        assert "pos_encoding" in dict(enc.named_buffers()), (
            "temporal PE must be a (frozen) buffer"
        )
        pe = enc.pos_encoding
        assert not pe.requires_grad, (
            "temporal PE must be frozen (REF: requires_grad_(False))"
        )
        std = pe.std().item()
        assert abs(std - _INIT_STD) < 0.004, (
            f"temporal PE std={std:.4f}, expected frozen-random ~{_INIT_STD}. "
            f"A sinusoidal table has std ~0.68 (that was the divergence)."
        )


class TestEmbedderInitParity:
    """REFERENCE WhisperEmbedder day-projection: day_As = torch.randn(...) (unscaled, std~1.0),
    day_Bs = 0, month weights = identity."""

    def test_proj_A_is_unscaled_randn(self):
        emb = _embedder()
        std = torch.stack(list(emb.proj_A)).std().item()
        assert abs(std - 1.0) < 0.1, (
            f"proj_A std={std:.4f}, expected ~1.0 (REF unscaled randn). A *0.01 scaling makes "
            f"the low-rank per-day delta (A@B, B init 0) learn ~100x slower."
        )

    def test_proj_B_inits_to_zero(self):
        assert all((p == 0).all() for p in _embedder().proj_B), (
            "proj_B must init to zero"
        )

    def test_day_proj_params_are_independent(self):
        """proj_W/proj_A/etc must be ParameterList (one Parameter per day), not a stacked
        tensor: Adam applies weight_decay to any parameter with a non-None .grad, so a
        stacked tensor gets ALL days' weights shrunk on every step even when only one
        day appears in the batch (REF's per-day Parameters correctly get .grad=None for
        days absent from the batch, and Adam skips them)."""
        emb = _embedder()
        assert isinstance(emb.proj_A, nn.ParameterList)
        assert isinstance(emb.proj_B, nn.ParameterList)
        assert isinstance(emb.proj_b_day, nn.ParameterList)
        assert isinstance(emb.proj_W_month, nn.ParameterList)
        assert isinstance(emb.proj_b_month, nn.ParameterList)

        emb_full_rank = _embedder(rank=0)
        assert isinstance(emb_full_rank.proj_W, nn.ParameterList)
        assert isinstance(emb_full_rank.proj_b_day, nn.ParameterList)

        # The defining behavior, not just the container type: a day absent from the
        # forward pass must get grad=None, not a zero-but-present gradient.
        x = torch.randn(2, 512, 100)
        day_idx = torch.zeros(2, dtype=torch.long)  # only day 0 touched
        out = emb_full_rank._apply_day_proj(x, day_idx)
        out.sum().backward()
        assert emb_full_rank.proj_W[0].grad is not None
        assert emb_full_rank.proj_W[1].grad is None, (
            "an untouched day's gradient must be None so Adam's weight_decay skips it"
        )

    def test_month_weights_init_to_identity(self):
        assert torch.allclose(_embedder().proj_W_month[0], torch.eye(512))


class TestEmbedderDropoutParity:
    """REFERENCE Card conv path applies day_layer_dropout(0.2) after the day projection and
    Dropout(0.1) on the session encoding; embed_dropout(0.4) is patch-mode only."""

    def test_dropout_probabilities_match_reference(self):
        emb = _embedder()
        assert emb.day_layer_dropout.p == 0.2, (
            "day-layer dropout must be 0.2 (REF hardcoded)"
        )
        assert emb.session_dropout.p == 0.1, (
            "session dropout must be 0.1 (REF PositionalEncoding)"
        )

    def test_no_single_proj_dropout(self):
        emb = _embedder()
        assert not hasattr(emb, "proj_dropout"), (
            "the old single proj_dropout(0.4) applied the patch-mode value on the conv path; "
            "it must be replaced by day_layer_dropout(0.2)."
        )

    def test_conv_path_has_no_embed_dropout_applied(self):
        # embed_dropout exists only for patch mode; a conv embedder must not carry it.
        emb = _embedder()
        assert not hasattr(emb, "embed_dropout"), (
            "embed_dropout is patch-mode only, not conv"
        )

    def test_session_encoding_passes_through_dropout(self):
        """In train mode the session-encoding path must actually drop (p=0.1).

        Published-run layout: the session encoding lands on the FIRST half,
        and the last half stays untouched.
        """
        emb = _embedder().train()
        torch.manual_seed(0)
        x = torch.zeros(4, 1500, 384)
        day = torch.zeros(4, dtype=torch.long)
        out = emb._add_day_encoding(x, day)
        assert out[:, :, 192:].abs().sum() == 0, (
            "last half must stay untouched under the published-run layout"
        )
        # with p=0.1 some session-encoding entries on the first half are
        # zeroed; the survivors are scaled by 1/(1-p), so the first half is
        # not simply the raw encoding.
        assert out[:, :, :192].abs().sum() > 0
