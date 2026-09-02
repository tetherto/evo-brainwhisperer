# Design Notes

Non-obvious design decisions in BrainWhisperer: what was chosen, what was rejected, and why. 

---

## Data layer (`brainwhisperer/data/`)

1. **`NeuralBatch` is a typed dataclass, not a dict**  
`NeuralBatch` (`data/batch.py`) is the contract between the data pipeline and the model. Every field is a typed tensor with a documented shape and meaning. A simpler strategy (i.e., passing a plain dict through the DataLoader) would let a key typo or shape error surface only when the model tries to use the value, potentially hundreds of GPU-hours into a run. Dataclasses fail at construction, before any computation.

2. **Dataset registry**  
`data/registry.py` maps string names to dataset classes via a `@register_dataset` decorator. A new dataset registers itself on import; the training entry-point reads names from config and calls `get_dataset(name)`, avoiding explict declarations and hard-coded paths.
The rejected alternative was a single `get_data_loaders` function with a long `if/else` chain, but this approach requires editing logic to add a new participant, making every addition a potential regression source.

3. **Tokenizer injected into `NeuralCollator`**  
Since dataset classes return raw sentence strings and the collator converts them to token ids, `NeuralCollator` (`data/collate.py`) receives the tokenizer as a constructor argument. 
Injecting the tokenizer in the collator instead of datasets, keeps each dataset class independent of any particular language model. This way, extending the model with a different decoder would only require passing a different tokenizer to the collator, with no changes to the dataset code.

---

## Evaluation (`eval.py`)

4. **Checkpoint loading with raw PyTorch**  
PyTorch 2.6 changed the default of `torch.load` from `weights_only=False` to `weights_only=True`, which doesn't work with arbitrary Python objects.
Lightning checkpoints store more than a `state_dict`: they also pickle the full `hyper_parameters` blob, which in this case is the `cfg` OmegaConf `DictConfig` saved via `save_hyperparameters()`.  
Calling `BWLightningModule.load_from_checkpoint()` in 2.6+ triggers `torch.load` internally with the new default, hits the pickled `DictConfig`, and raises an `UnpicklingError`.  
To avoid this, the config is restored from `config.yaml` via an OmegaConf merge; checkpoints are loaded with `weights_only=False` explicitly; and only the `state_dict` is used.

---

## Distributed training (`brainwhisperer/training/lightning_module.py`)

5. **Batch partitioning: shared-seed modulo split**
`StratifiedDayBatchSampler` gives every rank the *same* seed, so all ranks advance through an identical deterministic batch sequence; each rank then yields only the batches whose global position satisfies `global_idx % world_size == rank`.
Seeding each rank's RNG with `seed + LOCAL_RANK` would not prevent two independent RNGs to occasionally draw the same indices from the same pool. In this scenario, DDP would average gradients computed on partially identical data, silently shrinking the effective batch size.  
Modulo partitioning over a shared sequence is disjoint by construction (every integer belongs to exactly one residue class) and needs no inter-process communication.

6. **Dropping batch-count remainder**
Per-rank batch count is `floor(total_batches / world_size)`, with any remainder dropped rather than padded or unevenly distributed.
This keeps every rank's batch count exactly equal, which synchronized DDP requires. Padding was rejected because it would need extra bookkeeping to keep dummy or duplicated batches from biasing the loss.

7. **Setting `find_unused_parameters=True`**  
`BrainWhisperer.forward()` passes `encoder_outputs` into the Whisper model, so it skips its own encoder: those parameters always have grad=None and are never in the graph. With find `unused_parameters=False` (which is Lightning's DDPStrategy default), the code stall waiting  for a gradients never fills.  
Setting `find_unused_parameters=True` when world_size > 1, re-runs the unused-parameter traversal every step avoiding it. The trade-off with respect to using `static_graph=True` is a small per-step traversal cost instead of a one-time cost.

8. **`Trainer(use_distributed_sampler=False)` for the custom batch sampler**
`StratifiedDayBatchSampler` doesn't subclass `torch.utils.data.BatchSampler`, so Lightning's default attempt to inject a `DistributedSampler` into it fails at startup with a `TypeError`.
Avoiding the distributed sampled fixes it, but it has a side effect: Lightning also leaves the validation DataLoaders unsharded, so all ranks now validate on the full set redundantly rather than splitting the work. This redundancy is accepted here since the validation set is small and only checked once per epoch; if it becomes a bottleneck, the fix would be to shard validation explicitly with a `DistributedSampler` in `train.py` and reinstate the cross-rank gather.
