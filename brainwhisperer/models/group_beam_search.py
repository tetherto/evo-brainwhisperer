# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diverse (grouped) beam search, as built into transformers before 4.56.

transformers 5 moved it to a Hub repo that runs only with
`trust_remote_code=True`. This is a copy of that repo's code, so decoding
runs nothing downloaded:
https://huggingface.co/transformers-community/group-beam-search
at revision 56d0e48dadff53cb8b8653981b10534c50707fb1.

Changes from the original: its custom_generate/generate.py and
custom_generate/beam_search.py merged into one file, the `generate` wrapper
and the abstract `BeamScorer` base class removed, docstrings shortened, and
code reformatted. The decoding logic is unchanged.

Use: `whisper.generate(..., custom_generate=group_beam_search)`.
"""

from collections import UserDict

import torch
from torch import nn
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
)


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """Penalises tokens that earlier beam groups chose at the same step.

    See Diverse Beam Search (https://huggingface.co/papers/1610.02424).
    `diversity_penalty` is subtracted from a token's score once for each
    beam in an earlier group that chose that token at this step.
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError(
                "`diversity_penalty` should be a float strictly larger than 0."
            )
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError(
                "`num_beam_groups` should be an integer strictly larger than 1."
            )
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        """
        scores:          (batch_size * group_size, vocab_size) log-probabilities
        current_tokens:  (batch_size * num_beams) tokens chosen at this step
        beam_group_idx:  index of the group being processed
        """
        # hamming diversity: penalise using same token in current group which was
        # used in previous groups at the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        scores_processed = scores.clone()
        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams
                + group_start_idx
            ]
            token_frequency = torch.bincount(
                previous_group_tokens, minlength=vocab_size
            ).to(scores.device)
            scores_processed[batch_idx * group_size : (batch_idx + 1) * group_size] -= (
                self._diversity_penalty * token_frequency
            )

        return scores_processed


class BeamHypotheses:
    """The n best finished hypotheses of one beam group."""

    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        max_length: int | None = None,
    ):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be "
                "defined. Ensure it is passed to the BeamScorer class instance at "
                "initialization time."
            )

    def __len__(self):
        return len(self.beams)

    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: torch.LongTensor | None = None,
        generated_len: int | None = None,
    ):
        """Add a new hypothesis to the list."""
        if generated_len is not None:
            score = sum_logprobs / (generated_len**self.length_penalty)
        # This 'else' case exists for retrocompatibility
        else:
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(
        self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: int | None = 0
    ) -> bool:
        """True once no unfinished hypothesis can beat the worst kept one."""
        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even
        # though it is not entirely accurate when `length_penalty` is positive. See
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = (
                best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
            )
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of
        # `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtained from
            # `max_length`, not from `cur_len` -> min abs(`highest_attainable_score`)
            # is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError("max_length is not larger than decoder prompt length")
                highest_attainable_score = (
                    best_sum_logprobs
                    / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # the opposite logic applies here (max `highest_attainable_score` from
            # `cur_len`)
            else:
                highest_attainable_score = (
                    best_sum_logprobs
                    / (cur_len - decoder_prompt_len) ** self.length_penalty
                )
            ret = self.worst_score >= highest_attainable_score
            return ret


class BeamSearchScorer:
    """Keeps the beams and finished hypotheses of every (batch item, group).

    Adapted in part from Facebook's XLM beam search code
    (https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529)
    and Ashwin Kalyan's DBS implementation
    (https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua).
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float | None = 1.0,
        do_early_stopping: bool | str | None = False,
        num_beam_hyps_to_keep: int | None = 1,
        num_beam_groups: int | None = 1,
        max_length: int | None = None,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th
        # group in the i-th mini-batch.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of
        # the beam_hyps of the j-th group in the i-th mini-batch is complete.
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)],
            dtype=torch.bool,
            device=self.device,
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is "
                f"{num_beams}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead."
            )

        if (
            not isinstance(num_beam_groups, int)
            or (num_beam_groups > num_beams)
            or (num_beams % num_beam_groups != 0)
        ):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                f"`num_beam_groups`, but is {num_beam_groups} with `num_beams` being "
                f"{num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: int | torch.Tensor | None = None,
        eos_token_id: int | list[int] | torch.Tensor | None = None,
        beam_indices: torch.LongTensor | None = None,
        group_index: int | None = 0,
        decoder_prompt_len: int | None = 0,
    ) -> dict[str, torch.Tensor]:
        """Move finished candidates to the hypotheses; pick the next beams.

        input_ids:     (batch_size * group_size, seq_len) this group's beams
        next_scores:   (batch_size, 2 * group_size) top candidate scores
        next_tokens:   (batch_size, 2 * group_size) their tokens
        next_indices:  (batch_size, 2 * group_size) the beam each extends

        Returns next_beam_scores, next_beam_tokens and next_beam_indices, each
        (batch_size * group_size).
        """
        # add up to the length which the next_scores is calculated on (including
        # decoder prompt)
        cur_len = input_ids.shape[-1] + 1
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if batch_size != (input_ids.shape[0] // self.group_size):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, "
                    f"but a group beam size of {self.group_size} is expected by the "
                    "beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a "
                    f"beam size of {self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros(
            (batch_size, self.group_size), dtype=next_scores.dtype, device=device
        )
        next_beam_tokens = torch.zeros(
            (batch_size, self.group_size), dtype=next_tokens.dtype, device=device
        )
        next_beam_indices = torch.zeros(
            (batch_size, self.group_size), dtype=next_indices.dtype, device=device
        )

        if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(
                        f"Batch can only be done if at least {self.num_beams} beams "
                        "have been generated"
                    )
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError(
                        "Generated beams >= num_beams -> eos_token_id and pad_token "
                        "have to be defined"
                    )
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it
                    # should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.group_size
                    )
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                        generated_len=cur_len - decoder_prompt_len,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can "
                    f"be equal to `eos_token_id: {eos_token_id}`. Make sure "
                    f"{next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[
                batch_group_idx
            ] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: int | torch.Tensor | None = None,
        eos_token_id: int | list[int] | torch.Tensor | None = None,
        beam_indices: torch.LongTensor | None = None,
        decoder_prompt_len: int | None = 0,
    ) -> tuple[torch.LongTensor]:
        """Return the best `num_beam_hyps_to_keep` sequences per batch item.

        Returns sequences (batch_size * num_beam_hyps_to_keep, seq_len), padded
        and ended with EOS where they fit, plus sequence_scores and
        beam_indices.
        """
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = (
                    beam_indices[batch_beam_idx] if beam_indices is not None else None
                )
                generated_len = final_tokens.shape[-1] - decoder_prompt_len
                beam_hyp.add(
                    final_tokens,
                    final_score,
                    beam_indices=beam_index,
                    generated_len=generated_len,
                )

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[
                i * self.num_beam_groups : (i + 1) * self.num_beam_groups
            ]
            candidate_beams = [
                beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams
            ]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = (
            min(sent_lengths_max, max_length)
            if max_length is not None
            else sent_lengths_max
        )
        decoded: torch.LongTensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(
                batch_size * self.num_beam_hyps_to_keep, sent_max_len
            )
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


def group_beam_search(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    """Decoding loop for `generate(..., custom_generate=group_beam_search)`.

    Splits `num_beams` into `num_beam_groups` groups and runs beam search in
    each, penalising tokens earlier groups chose at the same step by
    `diversity_penalty`. `generate()` prepares every argument; for an
    encoder-decoder model, `model_kwargs` holds `encoder_outputs`.

    Returns the generated ids, or a GenerateBeam*Output if
    `return_dict_in_generate` is set.
    """
    # check parameters
    assert (
        generation_config.diversity_penalty != 0.0
        and generation_config.num_beam_groups != 1
    ), "Group beam search requires diversity_penalty > 0.0 and num_beam_groups > 1"
    if generation_config.do_sample is True:
        raise ValueError("Group beam search requires `do_sample` to be set to `False`")
    if generation_config.num_beams % generation_config.num_beam_groups != 0:
        raise ValueError(
            "Group beam search requires `num_beams` to be divisible by "
            "`num_beam_groups`"
        )
    if generation_config.diversity_penalty == 0.0:
        raise ValueError(
            "Group beam search requires `diversity_penalty` to be greater than `0.0`, "
            "otherwise your groups will be identical."
        )

    if streamer is not None:
        raise ValueError("Group beam search does not support streaming")

    if (
        generation_config.diversity_penalty is not None
        and generation_config.diversity_penalty > 0.0
    ):
        logits_processor.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )

    # define beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=input_ids.shape[0] // generation_config.num_beams,
        num_beams=generation_config.num_beams,
        device=input_ids.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        num_beam_groups=generation_config.num_beam_groups,
        max_length=generation_config.max_length,
    )
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
    device = input_ids.device

    batch_beam_size, cur_len = input_ids.shape
    # Does not exist anymore in recent versions!
    if hasattr(model, "_get_initial_cache_position"):
        model_kwargs = model._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

    if return_dict_in_generate and output_scores:
        beam_indices = [
            tuple(() for _ in range(num_sub_beams * batch_size))
            for _ in range(num_beam_groups)
        ]
    else:
        beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but "
            f"is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden
    # states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam of each group with 0 and the rest with -1e9.
    # This ensures that the beams in the same group don't produce same tokens every
    # time.
    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    decoder_prompt_len = input_ids.shape[1]  # record the prompt length of decoder
    while model._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # predicted tokens in cur_len step
        current_tokens = torch.zeros(
            batch_size * num_beams, dtype=input_ids.dtype, device=device
        )

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(
            batch_size * num_beams, dtype=torch.long, device=device
        )

        # in cached decoding, Transformers v5 expects only the latest token
        # after the first step
        is_first_iteration = cur_len == decoder_prompt_len
        next_sequence_length = (
            None
            if is_first_iteration or not model_kwargs.get("use_cache", True)
            else 1
        )

        # do one decoder step on all beams of all sentences in batch
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            is_first_iteration=is_first_iteration,
            **model_kwargs,
        )

        # prepare variable output controls (note: some models won't accept all
        # output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )

        outputs = model(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs
        # must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        if output_scores:
            processed_score = torch.zeros_like(outputs.logits[:, -1, :])
        if output_logits:
            # Copy is needed to avoid keeping a hanging ref to outputs.logits which
            # may be very large for first iteration (the clone itself is always
            # small)
            raw_logit_score = outputs.logits[:, -1, :].to(
                copy=True, device=input_ids.device
            )

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [
                        batch_idx * num_beams + idx
                        for idx in range(group_start_idx, group_end_idx)
                    ]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of current group only
            # No need to clone() the logits here as they will not retain
            # outputs.logits at the end of the loop
            # .float() is needed to retain precision for later logits manipulations
            next_token_logits = outputs.logits[batch_group_indices, -1, :].to(
                dtype=torch.float32, device=input_ids.device
            )

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * group_size, vocab_size)
            vocab_size = next_token_scores.shape[-1]

            next_token_scores_processed = logits_processor(
                group_input_ids,
                next_token_scores,
                current_tokens=current_tokens,
                beam_group_idx=beam_group_idx,
            )
            next_token_scores = next_token_scores_processed + beam_scores[
                batch_group_indices
            ].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            if output_scores:
                processed_score[batch_group_indices] = next_token_scores_processed.to(
                    processed_score.dtype
                )

            # reshape for beam search
            next_token_scores = next_token_scores.view(
                batch_size, group_size * vocab_size
            )

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at
            # least 1 non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                max(2, 1 + n_eos_tokens) * group_size,
                dim=1,
                largest=True,
                sorted=True,
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = (
                sum(beam_indices, ()) if beam_indices is not None else None
            )
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
                group_index=beam_group_idx,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if return_dict_in_generate and output_scores:
                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices[0]))
                )

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat(
                [group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            current_tokens[batch_group_indices] = group_input_ids[:, -1]

            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                + group_start_idx
                + (beam_idx % group_size)
            )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (processed_score,)
            if output_logits:
                raw_logits += (raw_logit_score,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if model.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        # This is needed to properly delete outputs.logits which may be very large
        # for first iteration. Otherwise a reference to outputs is kept which keeps
        # the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to
        # _reorder_cache() to save the maximum memory (that way the memory peak
        # does not include outputs.logits)
        del outputs

        # NOTE: we need to check if `model._reorder_cache` exists for special models
        # like RAG, RecurrentGemma etc.
        if model_kwargs.get("past_key_values", None) is not None:
            if hasattr(model, "_reorder_cache"):
                model_kwargs["past_key_values"] = model._reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )
            else:
                model_kwargs["past_key_values"].reorder_cache(reordering_indices)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if model.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
