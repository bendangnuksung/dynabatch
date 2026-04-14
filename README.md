# dynabatch

`dynabatch` is a drop-in batching utility for variable-length text generation workloads. It first does **Max Token Sampler/Batching** by sorting inputs by length, then adds a pre-trained **regressor** on top to increase batch size on shorter examples while keeping memory pressure relative to the first, hardest batch.

It is mainly built and tested for encoder-decoder machine translation style workloads, where input length is a decent proxy for output length and memory usage.

## Installation

```bash
pip install dynabatch
```

## When dynabatch helps

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

## Quick Start

This example shows the API shape, but to actually see dynabatch do something useful, `texts` should usually be a fairly large collection with varied sequence lengths. With only a few short examples, it will behave correctly, but you will not really see the throughput benefit.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dynabatch import build_dynamic_batch_dataloader

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").cuda()
device = torch.device("cuda")

texts = [
    "Hello world",
    "A much longer sentence that will tokenize to more input tokens.",
    "Short one",
]

# In real use, `texts` should typically contain many examples with varied lengths.

dataloader = build_dynamic_batch_dataloader(
    texts=texts,
    tokenizer=tokenizer,
    batch_size=32,
    max_input_token_length=256,
)

with torch.inference_mode():
    for batch in dataloader:
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
```

## More Examples

### Compare dynamic vs static batching

Use the same baseline `batch_size`, then toggle `dynamic_batch_mode`.

```python
dynamic_loader = build_dynamic_batch_dataloader(
    texts=texts,
    tokenizer=tokenizer,
    batch_size=32,
    max_input_token_length=256,
    dynamic_batch_mode=True,
)

static_loader = build_dynamic_batch_dataloader(
    texts=texts,
    tokenizer=tokenizer,
    batch_size=32,
    max_input_token_length=256,
    dynamic_batch_mode=False,
)
```

`dynamic_batch_mode=False` behaves like Max Token Sampler/Batching without the regressor-driven dynamic resizing. In other words, dynabatch is:

- Max Token Sampler/Batching
- plus optional dynamic batch growth on top

That makes `dynamic_batch_mode=False` useful as an ablation or sanity check.

### OOM-safe generation with fallback splitting

The regressor is empirical, so it can still occasionally predict a batch size that turns out too aggressive for a specific model, prompt template, GPU state, or generation setting. `generate_with_oom_fallback()` lets you keep the run alive by splitting only the failed batch into smaller chunks.

```python
import torch
from dynabatch import build_dynamic_batch_dataloader, generate_with_oom_fallback

device = torch.device("cuda")

dataloader = build_dynamic_batch_dataloader(
    texts=texts,
    tokenizer=tokenizer,
    batch_size=32,
    max_input_token_length=256,
)

with torch.inference_mode():
    for batch in dataloader:
        generated_tokens, did_fallback = generate_with_oom_fallback(
            model,
            batch,
            min_batch_size=32,
            device=device,
            max_new_tokens=128,
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
train_loader = build_dynamic_batch_dataloader(
    texts=texts,
    tokenizer=tokenizer,
    batch_size=16,
    max_input_token_length=256,
    friendly_batch_size=True,
    shuffle=True,
    shuffle_keep_first_n=3,
)
```

## How It Works

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

## API

### `build_dynamic_batch_dataloader`

```python
build_dynamic_batch_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_token_length: int = 512,
    threshold: float = 0.6,
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
    apply_template_func: Callable | None = None,
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
| `num_workers` | Worker count used by the returned `DataLoader`. If set to `0` or `None` outside debug mode, the implementation falls back to CPU count. |
| `debug` | Disables worker parallelism in the final loader to make debugging easier. |
| `dynamic_batch_mode` | If `True`, uses the regressor to vary batch size. If `False`, the loader reduces to Max Token Sampler/Batching with fixed batch size. This is the main switch for testing whether the dynamic part is actually helping your workload. |
| `smooth_batches` | If `True`, applies a smoothing pass after dynamic sizing so adjacent batches do not jump too abruptly in size. |
| `smooth_batches_max_diff` | Controls the largest allowed growth between adjacent batches as a fraction of `batch_size`. Example: `0.2` allows at most `0.2 * batch_size` extra items per step (still bounded by max batch size). |
| `apply_template_func` | Optional function applied to the batch texts before final tokenization. If your template adds tokens, make sure `max_input_token_length` still makes sense after templating. |
| `**tokenizer_kwargs` | Extra keyword arguments forwarded to the tokenizer during collation. |

The returned `DataLoader` yields dictionaries containing `input_ids`, `attention_mask`, `texts`, and any other tokenizer outputs.

## Regressor Training

The training pipeline and notebook notes now live in `train_regressor/readme.md`.

In short:

- the training data stores real GPU memory usage from many batch configurations
- the target is memory usage relative to the first batch
- the notebook trains an `XGBRegressor` to predict that ratio from token, word, and character statistics of the baseline batch and candidate batch

## Notebooks

- Inference comparison notebook: `notebooks/dynabatch_inference_comparison.ipynb`
- Regressor training notebook: `train_regressor/train_regression.ipynb`

Some older notebook cells may still show stale argument names or older thresholds, so prefer the current Python API in `dynabatch/main.py` for exact runtime behavior.
