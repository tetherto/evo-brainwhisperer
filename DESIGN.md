# Design Notes

Non-obvious design decisions in BrainWhisperer: what was chosen, what was rejected, and why. 

---

## Data layer (`brainwhisperer/data/`)

1. **`NeuralBatch` is a typed dataclass, not a dict**  
`NeuralBatch` (`data/batch.py`) is the contract between the data pipeline and the model. Every field is a typed tensor with a documented shape and meaning. A simpler strategy (i.e., passing a plain dict through the DataLoader) would let a key typo or shape error surface only when the model tries to use the value, potentially hundreds of GPU-hours into a run. Dataclasses fail at construction, before any computation.

2. **Dataset registry: one explicit mapping of stateless recipes**  
`data/recipes/__init__.py` exposes `RECIPES: dict[str, Recipe]`, one stateless instance per dataset. Each recipe builds its own datasets (`build_train_dataset`, `build_val_dataset`, `build_chain`) and carries the model-shape metadata the entry points need: `NUM_DAYS` (day-axis size of its embedder), `CHANNEL_FILL` (how to widen it to 512 channels) and `calendar_days()`. `train.py`, `eval.py` and `eval_lightbeam.py` reach every dataset through this one mapping, so no entry point hard-codes a path or carries a per-dataset branch.
The rejected alternative was a single `get_data_loaders` function with a long `if/else` chain, which requires editing shared logic to add a participant and makes every addition a potential regression source. Registration is now a plain dict literal rather than an import side-effect decorator: the explicit form removes any dependence on import order and keeps `data.datasets` checkable against the model, which `train.py` does with `assert len(cfg.data.datasets) == cfg.model.n_embedders`.
`build_in_parallel()` in the same module threads the six builds, because dataset preparation dominates start-up.

3. **Token ids come from the recipe, not from the collator**  
`Recipe.segmenter()` (`data/recipes/base.py`) wires the label extractors: `WhisperLabelEncoder` for the WHISPER field (BOS stripped, truncated at `data.max_seq_len`) and `PhonemeEncoder` for PHONEMES, both triggered on `type == 'Sentence'`. Token ids are therefore produced once, while the dataset is built, and travel inside each sample. `MixedRecipeCollator` and `SingleRecipeCollator` (`data/collate.py`) take no tokenizer at all: they widen channels and pad via `default_ragged_collator()`, whose `PadSpec` uses 0 for NEURAL and PHONEMES and -100, the cross-entropy ignore index, for WHISPER.
The tokenizer is lazy-loaded inside the extractor from a `whisper_name` field defaulting to `openai/whisper-tiny.en`, so a recipe names the decoder vocabulary instead of being handed an object. This keeps the data layer independent of any particular language model: extending the pipeline to a different decoder means changing one extractor field, with no edits to the collators or to the dataset code. The only place a tokenizer object is injected is the decode side (`BWLightningModule`, `prediction.py`), where it is used solely to turn ids back into text.

---

## Evaluation (`eval.py`)

4. **Checkpoint loading with raw PyTorch**  
PyTorch 2.6 changed the default of `torch.load` from `weights_only=False` to `weights_only=True`, which doesn't work with arbitrary Python objects.
Lightning checkpoints store more than a `state_dict`: they also pickle the full `hyper_parameters` blob, which in this case is the `cfg` OmegaConf `DictConfig` saved via `save_hyperparameters()`.  
Calling `BWLightningModule.load_from_checkpoint()` in 2.6+ triggers `torch.load` internally with the new default, hits the pickled `DictConfig`, and raises an `UnpicklingError`.  
To avoid this, the config is restored from `config.yaml` via an OmegaConf merge; checkpoints are loaded with `weights_only=False` explicitly; and only the `state_dict` is used.

---

## Distributed training (`train.py` and `tether_scratch.dataloader.StratifiedBatchSampler`)

5. **Batch partitioning: shared-seed modulo split**
`StratifiedBatchSampler` (`tether_scratch/dataloader.py`, constructed in `train.py`) is given the *same* seed on every rank with `seeding="continuous"`, so all ranks advance through one identical deterministic batch sequence; each rank then yields only the batches whose global position satisfies `global_index % num_replicas == rank`. The divisibility this needs is asserted twice, in `train.py` and again inside the sampler: `batch_size` and `days_per_batch` must both be multiples of the world size.
Seeding each rank's RNG with `seed + LOCAL_RANK` would not prevent two independent RNGs to occasionally draw the same indices from the same pool. In this scenario, DDP would average gradients computed on partially identical data, silently shrinking the effective batch size.  
Modulo partitioning over a shared sequence is disjoint by construction (every integer belongs to exactly one residue class) and needs no inter-process communication.

6. **Dropping batch-count remainder**
Per-rank batch count is `len(dataset) // (batch_size // num_replicas) // num_replicas`, i.e. `floor(total_batches / world_size)`, with any remainder dropped rather than padded or unevenly distributed. The sampler returns that number from `__len__` and sets `num_batches = per_rank * num_replicas`, so every rank agrees on the epoch length.
This keeps every rank's batch count exactly equal, which synchronized DDP requires. Padding was rejected because it would need extra bookkeeping to keep dummy or duplicated batches from biasing the loss.

7. **Setting `find_unused_parameters=True`**  
`BrainWhisperer.forward()` passes `encoder_outputs` into the Whisper model, so it skips its own encoder: those parameters always have grad=None and are never in the graph. With find `unused_parameters=False` (which is Lightning's DDPStrategy default), the code stall waiting  for a gradients never fills.  
Setting `find_unused_parameters=True` when world_size > 1, re-runs the unused-parameter traversal every step avoiding it. The trade-off with respect to using `static_graph=True` is a small per-step traversal cost instead of a one-time cost.

8. **`Trainer(use_distributed_sampler=False)` for the custom batch sampler**
`StratifiedBatchSampler` doesn't subclass `torch.utils.data.BatchSampler` (it derives from `Sampler[list[int]]`), so Lightning's default attempt to inject a `DistributedSampler` into it fails at startup with a `TypeError`.
Avoiding the distributed sampled fixes it, but it has a side effect: Lightning also leaves the validation DataLoaders unsharded, so all ranks now validate on the full set redundantly rather than splitting the work. This redundancy is accepted here since the validation set is small and only checked once per epoch; if it becomes a bottleneck, the fix would be to shard validation explicitly with a `DistributedSampler` in `train.py` and reinstate the cross-rank gather.
