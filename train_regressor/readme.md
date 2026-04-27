# Regressor Training Notes

This folder contains the offline workflow used to build the memory-pressure regressor shipped with `dynabatch`.

At runtime, `dynabatch` does not run these notebooks. The package only loads the trained model from `dynabatch/models/regressor.ubj`. These notebooks explain how that model was created: first by collecting real GPU memory measurements from generation workloads, then by training an `XGBRegressor` to predict how memory-heavy a candidate batch will be compared with the first, hardest batch.

## Why a Regressor Is Needed

`dynabatch` starts like a max-token sampler:

1. Tokenize all input texts.
2. Sort samples from longest to shortest.
3. Put the longest samples in the first batch.

The first batch is intentionally treated as the hardest batch. If a user chooses `batch_size=32`, that means "32 of the longest examples should fit on my GPU." Later batches contain shorter examples, so they often have room for more than 32 samples.

The regressor answers this question:

> If the first batch had size `batch_size`, how memory-heavy would this later candidate batch be relative to that first batch?

The prediction target is a ratio:

```text
memory_usage_diff = current_batch_gpu_memory_usage / first_batch_gpu_memory_usage
```

So:

- `1.0` means about as memory-heavy as the first batch.
- `< 1.0` means lighter than the first batch.
- `> 1.0` means heavier than the first batch and more likely to OOM.

At runtime, `dynabatch` tries several candidate batch sizes and picks the largest one whose predicted ratio is below the configured `threshold`.

<br>

---
<br>

## Data Generation

The data generation notebook is `generate_training_data.ipynb`.

The goal of this notebook is to create parquet files containing real measurements from model generation. Each row is one text sample, but the important values are batch-level values repeated across the rows belonging to that batch.

### Source Texts

The notebook downloads English-side machine translation training data from the ALMA repository:

- German-English
- Czech-English
- Icelandic-English
- Russian-English

It then creates a larger synthetic-ish text pool by randomly joining sentences until each generated text falls into one of several word-length ranges. This is not synthetic text from a language model; it is real translation text stitched together to create a wider length distribution.

The current augmentation config creates examples across these approximate word-length bands:

- 1-10 words
- 10-50 words
- 50-100 words
- 100-200 words
- 200-500 words

This matters because the regressor needs to see many different sequence lengths, not just normal short translation sentences.

### Models Used for Measurement

The notebook contains loaders for:

- `facebook/nllb-200-distilled-600M`
- `Helsinki-NLP/opus-mt-en-de`
- `haoranxu/ALMA-7B`

The current `v2` generation path uses `USE_MODEL = "nllb"`.

Important current limitation: for `v2`, newly generated data from the Marian and ALMA paths appeared to hurt the regressor performance a lot, so those model paths were not included in `v2`. The exact reason has not been investigated yet. It may be due to different tokenizer behavior, model architecture, generation settings, quantization, padding side, or memory profile shape, but that is only a hypothesis. For new `v2`-style data generation, treat NLLB-generated data as the reliable path for now.

### Batch Configs

The notebook runs many combinations of:

- `start_batches`: the baseline first-batch size
- `max_input_lengths`: the tokenizer truncation length
- `sample`: how many texts to sample for that run
- `increase_chance`: how often the generator deliberately probes a slightly larger batch

For NLLB, the current config tests baseline batch sizes such as `1`, `4`, `8`, `64`, `128`, and `256`, with max input lengths `128`, `256`, and `512`.

Each config is also duplicated with `increase_chance = 0.0`. This gives both conservative runs and exploratory runs. The exploratory runs are useful because a model trained only on conservative batches would not see enough examples near the memory limit.

### What `generate()` Records

For each config, the notebook:

1. Computes token lengths with `dynabatch.compute_lengths()`.
2. Sorts the sampled texts by token length, longest first.
3. Runs the first successful batch with `start_batch`.
4. Treats that first successful batch as the memory baseline.
5. Uses a token-budget heuristic to choose later batch sizes as texts get shorter.
6. Runs `model.generate()` and records peak CUDA memory for every batch.
7. Recovers from CUDA OOM by reducing the candidate batch size.
8. Saves the measured data to parquet.

The most important saved columns are:

- `texts`: the input text.
- `token_length`: truncated token length for each text.
- `gpu_memory_usage`: measured peak CUDA memory used by the batch.
- `total_tokens`: total token count in the batch.
- `total_paddings`: number of padding positions in the tokenized batch.
- `inference_batch_size`: actual batch size used for that batch.
- `top_token_size`: padded sequence length of the batch.
- `first_gpu_memory_peak_usage`: memory usage of the first successful batch.
- `first_batch_size`: batch size of the first successful batch.
- `first_total_tokens`, `first_total_paddings`, `first_top_token_size`: baseline batch statistics.
- `start_batch`, `max_input_length`, `model_name`: config metadata.

The first batch is special because every later batch is compared against it. In production, `dynabatch` follows the same idea: the first batch contains the longest examples and defines the memory reference point.

### Current Data Versions

The existing training data is organized into `v1` and `v2` data folders under `train_regressor/data`.

Some transparency about these versions:

- `v1` was generated earlier while the project was still changing. It includes ALMA and Marian generated data that is still good to use for training, but the current `generate_training_data.ipynb` config and `generate()` logic are not guaranteed to reproduce it exactly.
- `v2` is closer to the current NLLB-only generation path.
- The current training notebook uses the already generated `v1` and `v2` parquet data to train the model.
- Do not regenerate new Marian or ALMA data for `v2`-style training and mix it in until the performance issue is understood.

In other words, the already generated `v1` and `v2` files are still usable for training, but the current notebook should not be treated as a perfect recipe for reproducing every historical file.

<br>

---
<br>

## Training the Regressor

The training notebook is `train_regressor.ipynb`.

Its job is to turn the raw generation logs into a supervised regression dataset.

### Batch Grouping

The generation data stores one row per text, but the regressor predicts batch-level memory pressure. The notebook reconstructs batch groups using `inference_batch_size`.

For each batch group, it computes summary statistics for the candidate batch, called `y` features. It also computes the same kind of statistics for the first baseline batch, called `x` features.

The names use this convention:

- `_x`: baseline first batch
- `_y`: candidate later batch
- `_diff`: candidate value divided by baseline value

Example:

```text
token_sum_diff = token_sum_y / token_sum_x
batch_size_diff = batch_size_y / batch_size_x
```

### Target

The target column is:

```text
memory_usage_diff = gpu_memory_usage / first_gpu_memory_peak_usage
```

This makes the model predict relative pressure, not absolute memory in MB. That is important because absolute memory depends heavily on GPU, model, CUDA state, precision, and framework details. The relative target is more portable: "how big is this candidate compared with the known first batch?"

Failed OOM batches are removed before training:

```text
gpu_memory_usage > 0
```

The notebook also removes duplicate rows over a small feature subset so repeated equivalent examples do not dominate training.

### Features Used

The final feature set used in the notebook is:

```text
batch_size_x
batch_size_y
batch_size_diff
token_max_diff
token_mean_diff
token_sum_diff
word_max_diff
word_mean_diff
word_sum_diff
char_sum_diff
```

These features describe how the candidate batch differs from the baseline batch:

- `batch_size_diff` captures how much larger or smaller the candidate batch is.
- Token features capture the model-facing sequence length pressure.
- Word and character features give extra information about the raw text shape after truncation.
- Sum features are especially important because total work often scales with batch size times sequence length.

The package code in `dynabatch/regressor.py` recreates this same feature schema at runtime before calling the trained model.

### Train/Validation Split

The notebook builds stratification groups from `memory_usage_diff` so the validation split keeps examples across low, medium, and risky memory regions.

It also applies sample weights by target range. This is useful because the model is not equally sensitive everywhere:

- Very small ratios matter because they tell the sampler it can safely increase batch size.
- Ratios near `1.0` matter because that is the OOM boundary.
- Ratios above `1.0` matter because underpredicting them can lead to an unsafe batch.

The current model is an `xgboost.XGBRegressor` with squared-error regression.

### Evaluation Focus

The notebook reports normal regression metrics such as RMSE and R2, but the most practical check is whether the model avoids unsafe underprediction near the boundary.

The helper `accuracy_will_not_OOM()` checks cases where:

```text
actual memory_usage_diff > 1.0
predicted memory_usage_diff <= 1.0
```

Those are dangerous because the model would say a candidate is safe even though the measured data says it is heavier than the baseline.

For `dynabatch`, this kind of error is more important than a normal small regression error far away from the boundary.

## How the Trained Model Is Used in `dynabatch`

The production path is:

1. User calls `dynabatch_sampler(...)` or `build_dynabatch_dataloader(...)`.
2. `compute_lengths()` computes token, word, and character lengths for all texts.
3. `DynaBatchSampler` sorts samples by token length, longest first.
4. The first batch uses exactly the user-provided `batch_size`.
5. `build_baseline_features()` builds `_x` features from that first batch.
6. For each later position, `select_optimal_batch_size()` builds candidate `_y` features for several possible batch sizes.
7. The trained `XGBRegressor` predicts `memory_usage_diff` for each candidate.
8. The sampler chooses the largest candidate whose prediction is less than or equal to `threshold`.
9. Optional smoothing and batch-size rounding are applied.

The model file is loaded lazily from:

```text
dynabatch/models/regressor.ubj
```

The default `threshold` is below `1.0`, which leaves a safety margin. Lower thresholds are more conservative. Higher thresholds allow larger batches but increase OOM risk.

## Mental Model

A useful way to think about the system:

```text
First batch:
  "This is the hardest batch the user says can fit."

Later candidate batch:
  "Can I increase the number of shorter examples while staying below the first batch's memory pressure?"

Regressor:
  "Based on measured generation data, this candidate looks like 0.74x / 0.95x / 1.12x the first batch."
```

Then `dynabatch` picks the largest candidate that stays under the threshold.

## Practical Notes for Future Retraining

- Keep generated data versioned. The exact generation config matters.
- Do not mix data from new model families until their effect on validation behavior is understood.
- Pay extra attention to predictions around `0.8` to `1.2`, because this is where batch-size decisions are most sensitive.
- After training a new model, compare it against the current production model using `save_input_features=True` in `dynabatch_sampler`.
- Only replace `dynabatch/models/regressor.ubj` after checking that the new model behaves sensibly on production-like sampler inputs, not just on random validation rows.
