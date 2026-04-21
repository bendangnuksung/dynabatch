# dynabatch
[![PyPI version](https://img.shields.io/pypi/v/dynabatch.svg)](https://pypi.org/project/dynabatch/)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://pypi.org/project/dynabatch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/bendangnuksung/dynabatch/actions/workflows/test.yml/badge.svg)](https://github.com/bendangnuksung/dynabatch/actions/workflows/test.yml)

`dynabatch` is a drop-in batching utility for variable-length text generation workloads. It first does **Max Token Sampler/Batching** by sorting inputs by length, then adds a pre-trained **regressor** on top to increase batch size on shorter examples while keeping memory pressure relative to the first, hardest batch.

It is mainly built and tested for encoder-decoder machine translation style workloads, where input length is a decent proxy for output length and memory usage.

**Throughput:** In example [Notebooks](#notebooks), **inference generate()** on a T4 was **~1.06–1.21×** vs max-token sampling alone (three models). **Training** on an RTX 5090 was **~3×** vs fixed-batch training. The reason is simple: long examples force a conservative fixed batch size, which leaves memory headroom and compute underused on later shorter examples; dynabatch recovers part of that headroom. **Illustrative only.**

## 📥  Installation

```bash
pip install dynabatch
```

## ⚡ When dynabatch helps

dynabatch is most useful when:

- long examples force you to choose a conservative fixed batch size
- that conservative batch size leaves GPU compute underutilized on the many shorter examples later in the dataset
- your task has a reasonably predictable relation between input length and generation cost

It is generally a better fit for **encoder-decoder** models than for decoder-only LLMs. For decoder-only training or inference, sequence packing is often the stronger optimization because it reduces padding waste by filling token slots directly inside packed sequences. dynabatch can still help on decoder-only workloads in some cases, but it is not where I would position the library first.

This is the common translation scenario:

- a few very long inputs force a small safe batch size because they eat a lot of VRAM
- once those hard batches are out of the way, later shorter batches could fit many more examples
- increasing batch size there improves throughput and reduces wasted padding

It is less useful when the GPU is already compute-bound even at the smallest safe batch size. In that case, making the batch larger does not buy much. If you want to check that, compare:

- `dynamic_batch_mode=True`
- `dynamic_batch_mode=False`

If both behave similarly, dynabatch is probably not your bottleneck.

## ▶️ Quick Start

`dynabatch_sampler` is a **batch sampler**: use `DataLoader(..., batch_sampler=sampler)` (omit `batch_size`). The snippet below is copy-paste runnable.

```python
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dynabatch import dynabatch_sampler

texts = [
    "Hello world!",
    "This is a slightly longer example sentence for batching.",
    "Short one",
    "A much longer sentence is useful to create variable sequence lengths for testing dynabatch quickly.",
    "A much longer sentence is useful to create variable sequence lengths for testing dynabatch quickly, A much longer sentence is useful to create variable sequence lengths for testing dynabatch quickly",
    "Another medium-length sample.",
    "Tiny",
]

dataset = Dataset.from_dict({"text": texts})
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def collate_fn(batch):
    batch_texts = [x["text"] for x in batch]
    return tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

sampler = dynabatch_sampler(texts, tokenizer, batch_size=1, max_input_token_length=64)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

for i, batch in enumerate(loader):
    print(f"Batch No: {i} \t|\t Batch size: {len(batch["input_ids"])}")
```

Or use **`build_dynabatch_dataloader(texts, tokenizer, batch_size=1, max_input_token_length=64)`** for a built-in loader.

## 📒Notebooks 
[All Notebooks](./notebooks/)

| Notebook | Comparison | Notes/Observations |
|---|---|---|
| **Inference**<br>[🟠🟡 Colab](https://colab.research.google.com/github/bendangnuksung/dynabatch/blob/main/notebooks/dynabatch_inference_comparison.ipynb) | <img src="https://raw.githubusercontent.com/bendangnuksung/dynabatch/ed4c58a0deb2f03ec7d21aede1af2a9ce91cabbd/images/inference_generate_comparitive_analysis_table.png" alt="Inference comparison table" width="800"> | - Ran on Colab **T4**<br>- **~1.06×–1.21×** vs max-token sampler alone across three models (small gains)<br>- Bigger wins on heavy models (for example NLLB, Qwen): high memory use means smaller static batches, so dynamic batching helps more<br>- Faster GPUs at the same VRAM may see bigger gains than on a T4 |
| **Training**<br>[🟠🟡 Colab](https://colab.research.google.com/github/bendangnuksung/dynabatch/blob/main/notebooks/dynabatch_training_comparison.ipynb) | <img src="https://raw.githubusercontent.com/bendangnuksung/dynabatch/ed4c58a0deb2f03ec7d21aede1af2a9ce91cabbd/images/training_comparison.png" alt="Training comparison chart" width="800"> | - Ran on **RTX 5090**<br>- Roughly **3×** higher throughput vs standard Seq2Seq training with a fixed batch size<br>- Compares Hugging Face **Seq2SeqTrainer** (static batching) to **Dynabatch Trainer**<br>- Mostly **memory-bound** with a fixed batch: long examples force a small batch, so much of GPU compute sits idle on shorter sequences; dynabatch grows batches there<br>- Dynabatch can sometimes overestimate batch size and trigger OOM; the notebook shows OOM fallback that splits and retries smaller chunks |


## ➕ More Examples

### Compare dynamic vs static batching
```python
from torch.utils.data import DataLoader
from dynabatch import dynabatch_sampler

kw = dict(texts=texts, tokenizer=tokenizer, batch_size=32, max_input_token_length=256)
dynamic = DataLoader(
    dataset,
    batch_sampler=dynabatch_sampler(**kw, dynamic_batch_mode=True),
    collate_fn=collate_fn,
)
static = DataLoader(
    dataset,
    batch_sampler=dynabatch_sampler(**kw, dynamic_batch_mode=False),
    collate_fn=collate_fn,
)
```

`dynamic_batch_mode=False` behaves like Max Token Sampler/Batching without the regressor-driven dynamic resizing. In other words, dynabatch is:

- Max Token Sampler/Batching
- plus optional dynamic batch growth on top

That makes `dynamic_batch_mode=False` useful as a sanity check.

### OOM-safe generation with fallback splitting

The regressor is empirical, so it can still occasionally predict a batch size that turns out too aggressive for a specific model, prompt template, GPU state, or generation setting. `generate_with_oom_fallback()` lets you keep the run alive by splitting only the failed batch into smaller chunks.

```python
import torch
from torch.utils.data import DataLoader
from dynabatch import dynabatch_sampler, generate_with_oom_fallback

loader = DataLoader(
    dataset,
    batch_sampler=dynabatch_sampler(texts, tokenizer, batch_size=32, max_input_token_length=256),
    collate_fn=collate_fn,
)
device = torch.device("cuda")

with torch.inference_mode():
    for batch in loader:
        generated_tokens, did_fallback = generate_with_oom_fallback(
            model, batch, min_batch_size=32, device=device, max_new_tokens=128,
        )

        if did_fallback:
            print("Fallback path used for this batch after an OOM.")
```

This is useful when you want throughput from dynamic batching without letting one occasional OOM kill a long inference run.

### Training-style usage

For training:

- if you want hardware friendly sizes (`2^n` or `3 * 2^n`), enable `friendly_batch_size=True`
- if you want to avoid odd batch sizes, keep `keep_batch_size_even=True` (default)
- if you want shuffled batches, set `shuffle=True`
- `shuffle_keep_first_n=3` means the first 3 hardest batches stay unshuffled and only the later batches are shuffled
- keeping the earliest hardest batches fixed is useful because it lets you hit the worst memory cases early and find OOM problems sooner

```python
from torch.utils.data import DataLoader
from dynabatch import dynabatch_sampler

train_loader = DataLoader(
    dataset,
    batch_sampler=dynabatch_sampler(
        texts,
        tokenizer,
        batch_size=16,
        max_input_token_length=256,
        friendly_batch_size=True,
        shuffle=True,
        shuffle_keep_first_n=3,
    ),
    collate_fn=collate_fn,
)
```

### Hugging Face Trainer integration (plug-and-play)

If you train with Hugging Face `Trainer`, use the trainer helpers so you do not have to manually inject:

- `get_train_dataloader()` override for `batch_sampler`
- `compute_loss()` reweighting for variable micro-batch sizes under gradient accumulation

```python
from transformers import Seq2SeqTrainer # Seq2SeqTrainer  as
from dynabatch import (
    dynabatch_sampler,
    make_dynabatch_trainer,
    MemoryCleanupCallback,
)

sampler = dynabatch_sampler(
    texts=train_texts,
    tokenizer=tokenizer,
    batch_size=8,
    max_input_token_length=512,
    shuffle=True,
)

DynabatchSeq2SeqTrainer = make_dynabatch_trainer(Seq2SeqTrainer)

trainer = DynabatchSeq2SeqTrainer(
    dynabatch_sampler=sampler,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[MemoryCleanupCallback()],
)
```

When this is a good fit:

- You use a Hugging Face trainer class and want dynamic batching without custom trainer boilerplate.
- `gradient_accumulation_steps > 1` and micro-batch sizes vary by step.
- You want fairer baseline-vs-dynabatch comparisons where fewer dynabatch steps can otherwise change the effective LR schedule.

Why loss reweighting exists:

- With variable micro-batch sizes, a plain accumulated loss can bias optimizer updates toward steps with smaller/larger batch sizes.
- Dynabatch rescales each micro-batch by `current_batch_size / per_device_train_batch_size` so contribution is closer to sample-count-weighted behavior.

When to enable LR auto-scaling (Off by default):

- You are comparing against an older fixed-batch trainer/collator setup and want similar effective optimization signal per epoch.
- DynaBatch significantly reduces steps-per-epoch and you want a linear-scaling-style correction.

```python
trainer = DynabatchSeq2SeqTrainer(..., auto_scale_lr=True)
```

Power-user options:

- Use `DynabatchTrainerMixin` directly if you need custom multiple inheritance:
  `class MyTrainer(DynabatchTrainerMixin, Seq2SeqTrainer): ...`
- Use `scale_lr_for_dynabatch(args, sampler, dataset_size)` as a standalone helper if you want explicit LR control outside trainer construction.
- For non-text modalities, set `batch_size_key=...` (for example `"pixel_values"`) so batch-size extraction in `compute_loss()` reads the right tensor.

OOM fallback options:

- `oom_fallback="split_retry"`: on `torch.cuda.OutOfMemoryError` during `training_step`, retry the same step in smaller chunks and keep as many samples as possible.
- `oom_fallback="skip"`: clear memory and skip the failing step by returning a zero loss.
- `oom_fallback=None`: (default) disable fallback and re-raise OOM immediately.
- `oom_min_batch_size`: chunk size used by split-retry. If unset, the trainer uses `dynabatch_sampler.min_batch_size`.
- Every handled OOM increments an `oom_failed` counter and logs it via `Trainer.log(...)`, so it appears in Trainer/TQDM progress metrics.

```python
trainer = DynabatchSeq2SeqTrainer(
    ...,
    oom_fallback="split_retry",  # "split_retry" | "skip" | None
    oom_min_batch_size=2,
)
```


## ⚙️  API

### `dynabatch_sampler`

Returns `DynaBatchSampler` for `DataLoader(..., batch_sampler=sampler)`. Same sizing/shuffle kwargs as `build_dynabatch_dataloader`; `dataset` indices must match `texts`.

```python
dynabatch_sampler(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.65,
    max_batch_range: float = 2.0,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    keep_batch_size_even: bool = True,
    num_workers: int = 4,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    smooth_batches: bool = True,
    smooth_batches_max_diff: float = 0.2,
    token_lengths: list[int] | None = None,
    word_lengths: list[int] | None = None,
    char_lengths: list[int] | None = None,
) -> DynaBatchSampler
```

### `build_dynabatch_dataloader`

Same batching as `dynabatch_sampler`, returns a `DataLoader` with built-in collation; extra kwargs go to the tokenizer.

```python
build_dynabatch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.65,
    max_batch_range: float = 2.0,
    shuffle: bool = False,
    shuffle_seed: int = 21,
    shuffle_keep_first_n: int = 3,
    friendly_batch_size: bool = False,
    keep_batch_size_even: bool = True,
    num_workers: int = 4,
    debug: bool = False,
    dynamic_batch_mode: bool = True,
    smooth_batches: bool = True,
    smooth_batches_max_diff: float = 0.2,
    token_lengths: list[int] | None = None,
    word_lengths: list[int] | None = None,
    char_lengths: list[int] | None = None,
    **tokenizer_kwargs,
) -> DataLoader
```

| Parameter | Description |
|---|---|
| `texts` | Raw input strings. |
| `tokenizer` | Any tokenizer compatible with the Hugging Face tokenizer interface. |
| `batch_size` | The baseline batch size for the first, longest batch. In practice, set this to the largest safe batch size for your worst-case inputs. |
| `max_input_token_length` | Hard truncation limit used while estimating lengths and later tokenizing the batches. |
| `threshold` | Maximum allowed regressor prediction for a candidate batch. Roughly, `1.0` means "as memory-heavy as the first batch". Lower values are more conservative. |
| `max_batch_range` | Upper multiplier for candidate batch sizes relative to `batch_size`. With `batch_size=32` and `max_batch_range=2.0`, dynabatch will search up to about `64`. |
| `shuffle` | Shuffles the already-built batches. Within a batch, lengths stay similar. This is batch-level shuffling, not full random token-level mixing. |
| `shuffle_seed` | Seed used when shuffling. |
| `shuffle_keep_first_n` | Keeps the first few hardest batches in original order before shuffling the rest. For example, `3` means the first 3 longest/hardest batches remain fixed so you can detect early OOM issues quickly. |
| `friendly_batch_size` | Rounds chosen batch sizes down to hardware-friendly values such as powers of two or `3 * 2^n`. Useful for some training setups. |
| `keep_batch_size_even` | If `True`, rounds chosen batch sizes to even numbers. Enabled by default and useful for setups that prefer even per-step microbatch sizes. |
| `num_workers` | Worker count for the length pre-pass (`datasets.map`) and for the returned `DataLoader`. |
| `debug` | Disables parallel workers for the length pass and enables verbose sampler logging. |
| `dynamic_batch_mode` | If `True`, uses the regressor to vary batch size. If `False`, the loader reduces to Max Token Sampler/Batching with fixed batch size. This is the main switch for testing whether the dynamic part is actually helping your workload. |
| `smooth_batches` | If `True`, applies a smoothing pass after dynamic sizing so adjacent batches do not jump too abruptly in size. |
| `smooth_batches_max_diff` | Controls the largest allowed growth between adjacent batches as a fraction of `batch_size`. Example: `0.2` allows at most `0.2 * batch_size` extra items per step (still bounded by max batch size). |
| `token_lengths`, `word_lengths`, `char_lengths` | Optional precomputed lengths aligned with `texts`. Provide all three to skip the upfront `compute_lengths` tokenization pass. |
| `**tokenizer_kwargs` | Extra keyword arguments forwarded to the tokenizer during collation (for example `truncation=True`). |

The returned `DataLoader` yields dictionaries containing `input_ids`, `attention_mask`, `texts`, and any other tokenizer outputs.

### Trainer helpers

```python
make_dynabatch_trainer(trainer_cls: type) -> type
scale_lr_for_dynabatch(
    args: Any,
    sampler: DynaBatchSampler,
    dataset_size: int,
    baseline_batch_size: int | None = None,
) -> Any
class DynabatchTrainerMixin
```

- `make_dynabatch_trainer`: builds a cached subclass combining `DynabatchTrainerMixin` with your trainer class.
- `DynabatchTrainerMixin`: overrides train dataloader + loss reweighting for variable micro-batch sizes, and adds optional OOM fallback in `training_step`.
- `scale_lr_for_dynabatch`: standalone helper mainly for fair fixed-vs-dynabatch comparisons; keep it off for normal training unless you explicitly want step-count-based LR adjustment.


## 🛠️ How It Works

```text
Static batching (fixed size chosen for longest examples)
  Long seqs   -> [####....] [####....] [####....]
  Short seqs  -> [##......] [##......] [##......]
                 ^ lots of padded / underused slots

dynabatch (grow batch as sequences get shorter)
  Long seqs   -> [####....]
  Medium seqs -> [###.....] [###.....]
  Short seqs  -> [##......] [##......] [##......] [##......]
                 ^ denser utilization across the epoch
```

1. All texts are tokenized up front to estimate truncated token, word, and character lengths.
2. Samples are sorted by token length from longest to shortest. This part alone is essentially Max Token Sampler/Batching.
3. The first batch uses exactly `batch_size` items. This is the hardest batch and becomes the baseline.
4. For every later batch, dynabatch builds candidate batch sizes from `batch_size` up to `batch_size * max_batch_range`.
5. A pre-trained `XGBRegressor` predicts memory pressure for each candidate relative to the first batch.
6. dynabatch chooses the largest candidate whose predicted load is less than or equal to `threshold`.
7. If `dynamic_batch_mode=False`, step 5 and step 6 are skipped and the pipeline reduces to Max Token Sampler/Batching with fixed batch size.

The important intuition is:

- around `1.0` means "about as memory heavy as the first batch"
- below `1.0` means lighter than the first batch
- above `1.0` means heavier than the first batch and therefore riskier

So you should choose `batch_size` as the largest batch of your longest inputs that safely fits on your GPU. The regressor then tries to grow from there when the later inputs get shorter.


## Regressor Training

The training pipeline and notebook notes now live in `train_regressor/readme.md`.

In short:

- the training data stores real GPU memory usage from many batch configurations
- the target is memory usage relative to the first batch
- the notebook trains an `XGBRegressor` to predict that ratio from token, word, and character statistics of the baseline batch and candidate batch
